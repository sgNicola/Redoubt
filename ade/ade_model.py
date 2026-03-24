import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        # Backbone with improved initialization and regularization
        self.input_norm = nn.LayerNorm(input_dim)  #  nn.BatchNorm1d(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),  # Normalize hidden states
            nn.Dropout(dropout_rate),  # Regularization to prevent overfitting
            nn.Linear(hidden_dim, 128),
            nn.GELU()
        )
        
        # Output heads for different tasks
        self.ade_head = nn.Sequential(
            nn.Linear(128, 64),  # Add one more layer for better feature extraction
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.collision_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.drivable_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Apply custom weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Extract features using the backbone
        x = self.input_norm(x)  
        features = self.backbone(x)
        
        # Predict outputs for all heads
        return {
            "ade": self.ade_head(features).squeeze(-1),
            "collision": self.collision_head(features),   
            "drivable": self.drivable_head(features)     
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')  # Use ReLU gain  # He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)