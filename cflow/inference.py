import time
import torch
import pytorch_lightning as pl
from torch import nn, Tensor
# flow_matching
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import matplotlib.pyplot as plt
from matplotlib import cm
# To avoide meshgrid warning
import warnings
import os
from pathlib import Path
from torch.distributions import Independent, Normal
from torch.distributions import MixtureSameFamily, Categorical, Normal
from src.exp.experiment import setup_logger
# from cflow.flow_trainer import FlowMatchingLightningModule
from cflow.ablition_trainer import FlowMatchingLightningModule
from ade.ade_trainer import LitSparseRegression
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import numpy as np  
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, device="cuda"):
        self.root = root
        self.split = split
        self.device = device
        self.path = os.path.join(self.root, self.split)
        self.file_paths = [f for f in Path(
            self.path).rglob('*.pt') if f.is_file()]
        self.file_paths.sort()

    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        item_data = dict()
        # data["hidden_feature"].shape [T, original_bs, agents,feature_dim=128]
        item_data["hidden_feature"]= data["hidden_feature"][:,0, 0].clone().detach()
        item_data["scenario_name"]=self.file_paths[index].stem
        return item_data["hidden_feature"], item_data["scenario_name"]
    
    def __len__(self):
        return len(self.file_paths)
 
def flow_sample(
    model_checkpoint_path,  # Path to the saved model checkpoint
    dataloader, 
    step_size, 
):
    # Load the model from the checkpoint
    print(f"Loading model from checkpoint: {model_checkpoint_path}")
    flow_model = FlowMatchingLightningModule.load_from_checkpoint(model_checkpoint_path)
    flow_model.eval()  # Set the model to evaluation mode
    flow_model.to(device)  # Move model to GPU (or CPU) as needed

    # Wrap the flow model to use it in the solver
    wrapped_vf = WrappedModel(flow_model)
    # solver = ODESolver(velocity_model=wrapped_vf)
    gaussian_log_density = Independent(
        Normal(torch.zeros(128, device=device), torch.ones(128, device=device)), 1
    ).log_prob
    solver = ODESolver(velocity_model=wrapped_vf)   
    k = 5
    latent_dim = 128
    weights = Categorical(torch.ones(k, device=device))
    locs = torch.randn(k, latent_dim, device=device)
    scales = torch.ones(k, latent_dim, device=device)
    gmm_log_density = MixtureSameFamily(
    mixture_distribution=weights,
    component_distribution=Independent(Normal(locs, scales), 1)
    ).log_prob
    # Iterate through the dataloader
    log_density = {"scenario_name": [], "log_p": []}
    with torch.no_grad(): 
        for batch_idx, hidden_feature in enumerate(dataloader):
            features, scenario_name=hidden_feature
            x_1 = features[0].to(device)
            _, exact_log_p = solver.compute_likelihood(
            x_1=x_1,
            method='euler',
            step_size=step_size,
            atol=1e-4,     
            rtol=1e-4,    
            exact_divergence=True,
            log_p0=  gmm_log_density
            )
            log_density["scenario_name"].append(scenario_name[0])
            log_density["log_p"].append(exact_log_p.cpu().numpy().tolist())
    df = pd.DataFrame(log_density)
    return df


def run(data_dir, flow_checkpoint_path):
    proj_root = os.getcwd()
    step_size = 0.1
    save_dir = os.path.join(data_dir, "prediction")
    os.makedirs(save_dir, exist_ok=True)
    test_set = SimulationDataset(root=data_dir , split="simulation_results")    
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=64,
        pin_memory=True,
        persistent_workers=False,
    )
    
    df=flow_sample(flow_checkpoint_path, test_dataloader, step_size)
    save_path = f"{save_dir}/op_nopro_density.parquet"
    df.to_parquet(save_path, index=False)
    print("save successfully")
    
if __name__ == "__main__":
    data_dir = "/home/sgwang/PlanScope/pluto_dataset"
    flow_checkpoint_path = '/home/sgwang/PlanScope/FM_model/pluto_flow_nopro-epoch=1659-val_loss=0.3321.ckpt'
    run(data_dir,  flow_checkpoint_path)