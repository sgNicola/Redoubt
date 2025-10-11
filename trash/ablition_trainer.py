import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import time
import torch

from torch import nn, Tensor
# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

# visualization
import matplotlib.pyplot as plt
from matplotlib import cm
# To avoide meshgrid warning
import warnings
import os
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Independent, Normal
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
from cflow.flow_model import FlowMatchingModelNoProjection

class FlowMatchingLightningModule(pl.LightningModule):
    def __init__(self, feature_dim, hidden_dim, learning_rate):
        super().__init__()
        # Save hyperparameters (useful for logging or reproducibility)
        self.save_hyperparameters()

        # Define the flow model
        self.flow_model = FlowMatchingModelNoProjection(
            feature_dim=feature_dim, 
            hidden_dim=hidden_dim
        )

        # Define the learning rate
        self.learning_rate = learning_rate

        # Define the loss function (Smooth L1 Loss or MSE)
        # self.loss_fn = nn.SmoothL1Loss()
        # self.loss_fn = nn.MSELoss()

        # Path and scheduler
        self.path = AffineProbPath(scheduler=CondOTScheduler())

    def forward(self, x_t, t):
        """Forward pass for inference."""
        return self.flow_model(x_t, t)

    def training_step(self, batch, batch_idx):
        """Define the training loop logic."""
        hidden_feature = batch.to(self.device)
        t = torch.rand(hidden_feature.size(0)).to(self.device)
        x_0 = torch.randn_like(hidden_feature).to(self.device)

        # Sample path and compute predicted flow
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=hidden_feature)
        predicted_dx_t = self.flow_model(path_sample.x_t, path_sample.t)

        # Compute loss
        flow_loss =torch.pow(predicted_dx_t - path_sample.dx_t, 2).mean()
        self.log("train_loss", flow_loss, prog_bar=True, on_step=True, on_epoch=True)
        return flow_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation logic."""
        hidden_feature = batch.to(self.device)
        t = torch.rand(hidden_feature.size(0)).to(self.device)
        x_0 = torch.randn_like(hidden_feature).to(self.device)

        # Sample path and compute predicted flow
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=hidden_feature)
        predicted_dx_t = self.flow_model(path_sample.x_t, path_sample.t)

        # Compute validation loss
        val_loss = torch.pow(predicted_dx_t - path_sample.dx_t, 2).mean()
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]