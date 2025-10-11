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
import numpy as np

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim//2) * scale, requires_grad=False)

    def forward(self, t):
        if t.dim() == 0: 
            t = t.reshape(-1)
        t = t.view(-1, 1)
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        if t_embed.dim() == 3:
            t_embed = t_embed.squeeze(1)
        return t_embed
    
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            Swish(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.block(x)
    
    
class FlowMatchingModel(nn.Module):
    def __init__(self, feature_dim=128, time_embed_dim=128, hidden_dim=512):
        super().__init__()
        self.t_encoder = GaussianFourierProjection(time_embed_dim)
        self.main = nn.Sequential(
            nn.Linear(feature_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish(),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
        )
        for layer in self.main.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    
    def forward(self, features, t):
        if isinstance(t, (int, float)):
            t = torch.full((features.size(0),), t, device=features.device)
        elif t.dim() == 0:
             t = t.expand(features.size(0))
        t_embed = self.t_encoder(t)
        h = torch.cat([features, t_embed], dim=1)
        return self.main(h)

class FlowMatchingModelNoProjection(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=512):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish(),
            ResidualBlock(hidden_dim),
            # ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
        )
        for layer in self.main.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    
    def forward(self, features, t):
        # Ensure t is a 1D or scalar tensor and reshape it to (batch_size, 1)
        if isinstance(t, (int, float)):
            t = torch.full((features.size(0), 1), t, device=features.device)  # Shape: (batch_size, 1)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(features.size(0), 1)  # Shape: (batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)  # Shape: (batch_size, 1)

        # Concatenate features (batch_size, feature_dim) and t (batch_size, 1)
        h = torch.cat([features, t], dim=1)  # Shape: (batch_size, feature_dim + 1)
        return self.main(h)
    