import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import defaultdict
from ade.ade_model import SparseRegressionModel

class LitSparseRegression(pl.LightningModule):
    def __init__(self, input_dim, config, hidden_dim=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SparseRegressionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        self.config = config
        self.lr = lr
    
    def forward(self, x):
        """Forward pass for inference."""
        return self.model(x)
    
    def _compute_losses(self, outputs, targets):
        """
        Compute the total loss and individual task losses.
        
        Args:
            outputs (dict): Model predictions, with keys "ade", "collision", "drivable".
            targets (dict): Ground truth values, with keys "ADE", "collision_energy", "drivable_score".
        
        Returns:
            total_loss (torch.Tensor): The total weighted loss.
            loss_dict (dict): Dictionary of individual loss values.
        """
        loss_dict = {}
        device = outputs['ade'].device  # Get the device for consistency

        # ADE loss (Smooth L1 or MSE)
        loss_dict['ade'] = self.config['ade_weight'] * F.mse_loss(
            outputs['ade'], 
            targets['ADE']
        )
        
        # Initialize BCEWithLogitsLoss once, outside the loss computation loop
        collision_loss_fn = torch.nn.BCEWithLogitsLoss()
        drivable_loss_fn = torch.nn.BCEWithLogitsLoss()

        # Compute collision classification loss
        collision_mask = (targets['collision_energy'] > 0).float()  # Binary mask
        loss_dict['collision_cls'] = self.config['collision_cls_weight'] * collision_loss_fn(
            outputs['collision'].squeeze(-1),  # Raw logits, no sigmoid applied
            collision_mask
        )

        # Compute drivable classification loss
        drivable_mask = (targets['drivable_score'] > 0).float()  # Binary mask
        loss_dict['drivable_cls'] = self.config['drivable_cls_weight'] * drivable_loss_fn(
            outputs['drivable'].squeeze(-1),  # Raw logits, no sigmoid applied
            drivable_mask
        )

        
        ade_weight = 0.6
        collision_weight = 0.2
        drivable_weight = 0.2

        total_loss = (ade_weight * loss_dict['ade'] +
                    collision_weight * loss_dict['collision_cls'] +
                    drivable_weight * loss_dict['drivable_cls'])

        return total_loss, loss_dict
 
    def training_step(self, batch, batch_idx):
        x = batch["hidden_features"]
        targets = {
            "ADE": batch["ADE"],
            "collision_energy": batch["collision_energy"],
            "drivable_score": batch["drivable_score"]
        }
        outputs = self(x)
        total_loss, loss_dict = self._compute_losses(outputs, targets)

        metrics = {f"train_{k}": v for k, v in loss_dict.items()}
        metrics["train_loss"] = total_loss
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch["hidden_features"]
        targets = {
            "ADE": batch["ADE"],
            "collision_energy": batch["collision_energy"],
            "drivable_score": batch["drivable_score"]
        }
        outputs = self(x)
        total_loss, loss_dict = self._compute_losses(outputs, targets)

        metrics = {f"val_{k}": v for k, v in loss_dict.items()}
        metrics["val_loss"] = total_loss
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
