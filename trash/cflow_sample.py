import time
import torch
import pytorch_lightning as pl
from torch import nn, Tensor
# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.optim.lr_scheduler import StepLR
# To avoide meshgrid warning
import warnings
import os
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Independent, Normal
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from src.exp.experiment import setup_logger
from cflow.flow_trainer import FlowMatchingLightningModule

warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import numpy as np  
from cflow.train_cflow import HiddenDataset, custom_collate_fn


os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

def flow_sample(
    model_checkpoint_path,  # Path to the saved model checkpoint
    dataloader, 
    step_size, 
    batch_size, 
    num_acc=10
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
    with torch.no_grad(): 
        for batch_idx, hidden_feature in enumerate(dataloader):
            hidden_feature = hidden_feature.to(device)
            x_1 = hidden_feature
            _, exact_log_p = solver.compute_likelihood(
            x_1=x_1,
            method='euler',
            step_size=step_size,
            atol=1e-4,     
            rtol=1e-4,    
            exact_divergence=True,
            log_p0=gaussian_log_density
            )
            # Print results for the batch
            print(f"Batch {batch_idx + 1}, Exact Log Likelihood: {exact_log_p}")

def main():
        # Path to the saved model checkpoint
    data_dir = "/home/sgwang/PlanScope/pluto_dataset"
    model_checkpoint_path = "./FM_model/flow_model-epoch=29-train_loss=0.4034.ckpt"
    val_set = HiddenDataset(root= data_dir, split="val")
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1024, shuffle=True, num_workers=96, pin_memory=False, persistent_workers=False,  collate_fn = custom_collate_fn)
    # Assuming dataloader is already defined

    # Hyperparameters
    step_size = 0.1
    batch_size = 1024
    num_acc = 10  # Number of Hutchinson trace estimates

    # Call the function
    flow_sample(model_checkpoint_path, val_dataloader, step_size, batch_size, num_acc=num_acc)
    
if __name__ == "__main__":
    main()