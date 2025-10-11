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
from src.exp.experiment import setup_logger
from cflow.flow_trainer import FlowMatchingLightningModule
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import numpy as np  
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from ade.inference_ade import SimulationDataset
    
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

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
    solver = ODESolver(velocity_model=wrapped_vf)   
    # Define the Gaussian log density
    gaussian_log_density = Independent(
        Normal(torch.zeros(128, device=device), torch.ones(128, device=device)), 1
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
            log_p0=gaussian_log_density
            )
            log_density["scenario_name"].append(scenario_name[0])
            log_density["log_p"].append(exact_log_p.cpu().numpy().tolist())
            print(len(log_density["log_p"]))
            print((len(log_density['scenario_name'])))
            break
    #xxxxx： @TODO align with result reports
    df = pd.DataFrame(log_density)
    df.to_csv("log_density.csv", index=False)
  
def simulation():
     
    proj_root = os.getcwd()
    model_checkpoint_path = "./FM_model/flow_model-epoch=29-train_loss=0.4034.ckpt"
    test_set = SimulationDataset(root= proj_root, split="demo_simulation_results")    
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=64,
        pin_memory=True,
        persistent_workers=False,
    )
    
        # Hyperparameters
    step_size = 0.1
    # num_acc = 10  # Number of Hutchinson trace estimates
    # Call the function
    flow_sample(model_checkpoint_path, test_dataloader, step_size)

if __name__ == "__main__":
    simulation()