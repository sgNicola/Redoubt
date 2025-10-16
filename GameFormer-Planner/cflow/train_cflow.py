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
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import argparse
from exp.experiment import setup_logger
from cflow.flow_trainer import FlowMatchingLightningModule

warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import numpy as np  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class HiddenDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, device="cuda"):
        self.root = root
        self.split = split
        self.device = device
        self.path = os.path.join(self.root, self.split + "_results")
        self.file_paths = [f for f in Path(
            self.path).rglob('*.pt') if f.is_file()]
        self.file_paths.sort()

    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        item_data = dict()
        item_data["hidden_feature"] = data["hidden_feature"].clone().detach()
        return item_data["hidden_feature"]  # Return the full tensor [3, 128]
 

    def __len__(self):
        return len(self.file_paths)
    
def custom_collate_fn(batch):
    # Flatten each sample's chunks into individual tensors
    all_hidden_features = [chunk for sample in batch for chunk in torch.chunk(sample, 1, dim=0)]  # List of [1, 128]
    # Remove the extra dimension (from [1, 128] to [128])
    all_hidden_features = [chunk.squeeze(0) for chunk in all_hidden_features]
    return torch.stack(all_hidden_features)  # Combine into a single tensor

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

    
def train(resume, data_dir, model_name, ckpt_path=None):
    proj_root = os.getcwd()
    tb_logger, cmd_logger = setup_logger(proj_root)
    feature_dim = 512
    hidden_dim = 512
    learning_rate = 0.001    
    train_set = HiddenDataset(root= data_dir, split="train")
    val_set = HiddenDataset(root= data_dir, split="val")

    
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1024,
        shuffle=True,
        num_workers=64,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1024,
        shuffle=False,
        num_workers=64,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn
    )

    # ================== model ==================
    if resume and ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming training from checkpoint: {ckpt_path}")
        model = FlowMatchingLightningModule.load_from_checkpoint(
            ckpt_path,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate
        )
    else:
        print("Initializing new model")
        model = FlowMatchingLightningModule(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate
        )

    # ================== call back ==================
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./FM_model",
        filename=f"{model_name}_flow-{{epoch:03d}}-{{val_loss:.4f}}",
        save_top_k=3,
        mode="min",
        save_last=True,   
        every_n_epochs=10
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ================== trainer ==================
    trainer = pl.Trainer(
        max_epochs=2000,
        logger=tb_logger,
        devices='auto',
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed",
        gradient_clip_val=1.0
    )

    # ================== train ==================
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path if resume else None  
    )
 

    
if __name__ == "__main__":
    # ================== Argument Parser ==================
    parser=argparse.ArgumentParser(description="Train a model with specified options.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoint to resume training from.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (used for saving checkpoints).")

    args = parser.parse_args()
    # ckpt_path = "./FM_model/flow_model-epoch=892-train_loss=0.3676.ckpt"
    # train(resume=True, ckpt_path=ckpt_path)
    # train(resume=False,data_dir="/home/sgwang/PlanScope/plantf_dataset")
    # Call train function with parsed arguments
    train(
        resume=args.resume,
        data_dir=args.data_dir,
        model_name=args.model_name,
        ckpt_path=args.ckpt_path
    )