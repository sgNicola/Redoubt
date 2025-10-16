from ade.ade_trainer import LitSparseRegression
from ade_datamodule import HiddenDataset, custom_collate_fn
import pytorch_lightning as pl
import os
import torch
from exp.experiment import setup_logger
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import argparse

config = {
    'pos_weight': 2.0,
    'ade_weight': 1.0,
    'collision_cls_weight': 0.8,
    'drivable_cls_weight': 0.7,
}

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
        batch_size=32,
        shuffle=True,
        num_workers=64,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=64,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn
    )

    # ================== model ==================

    if resume and ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming training from checkpoint: {ckpt_path}")
        model = LitSparseRegression.load_from_checkpoint(
            ckpt_path,
            input_dim=feature_dim,
            config =config,
            hidden_dim=hidden_dim,
            lr=learning_rate
        )
    else:
        print("Initializing new model")
        model =  LitSparseRegression(
            input_dim=feature_dim,
            config =config,
            hidden_dim=hidden_dim,
            lr=learning_rate
        )

    # ================== call back ==================
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./ade_model",
        filename=f"{model_name}_flow-{{epoch:03d}}-{{val_loss:.4f}}",
        save_top_k=3,
        mode="min",
        save_last=True,   
        every_n_epochs=10
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ================== trainer ==================
    trainer = pl.Trainer(
        max_epochs=100,
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
    train(
        resume=args.resume,
        data_dir=args.data_dir,
        model_name=args.model_name,
        ckpt_path=args.ckpt_path
    )
 