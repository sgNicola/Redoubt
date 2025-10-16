import time
import torch
import pytorch_lightning as pl
from torch import nn, Tensor
# flow_matching

from flow_matching.utils import ModelWrapper
import matplotlib.pyplot as plt
from matplotlib import cm
# To avoide meshgrid warning
import warnings
import os
from pathlib import Path
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
        # data["hidden_feature"].shape [T, original_bs, agents,feature_dim]
        item_data["hidden_feature"]= data["hidden_feature"][:,0, :].clone().detach()
        item_data["scenario_name"]=self.file_paths[index].stem
        return item_data["hidden_feature"], item_data["scenario_name"]
    
    def __len__(self):
        return len(self.file_paths)
    

def driving_score_inference(
     model_checkpoint_path,
    dataloader, 
):
    print(f"Loading model from checkpoint: {model_checkpoint_path}")
    driving_score_model =LitSparseRegression.load_from_checkpoint(model_checkpoint_path)
    driving_score_model.eval()
    driving_score_model.to(device)
    driving_score_risk = {"scenario_name": [], "ade": [], "collision_risk":[], "drive_compliance":[]}
    with torch.no_grad():
        for batch_idx, hidden_feature in enumerate(dataloader):
            features, scenario_name = hidden_feature
            inputs = features[0].to(device)
            outputs = driving_score_model(inputs)
            driving_score_risk["scenario_name"].append(scenario_name[0])
            driving_score_risk["ade"].append(outputs["ade"].cpu().numpy().tolist())
            driving_score_risk["collision_risk"].append(outputs["collision"].cpu().numpy().tolist())
            driving_score_risk["drive_compliance"].append(outputs["drivable"].cpu().numpy().tolist())
            # print(outputs)
    df = pd.DataFrame(driving_score_risk)
    return df

def run(data_dir, ade_checkpoint_path):
    save_dir = os.path.join(data_dir, "prediction")
    os.makedirs(save_dir, exist_ok=True)
    test_set = SimulationDataset(root= data_dir, split="simulation_results")    
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=64,
        pin_memory=True,
        persistent_workers=False,
    )
 
    driving_risk= driving_score_inference(
        ade_checkpoint_path,
        test_dataloader
        
    )
    save_path = f"{save_dir}/driving_risk.parquet"
    driving_risk.to_parquet(save_path, index=False)
if __name__ == "__main__":
    data_dir = "/home/sgwang/GameFormer-Planner/gameformer_dataset"
    ade_checkpoint_path = 'ade_model/gameformer_flow-epoch=009-val_loss=0.6763.ckpt'
    run(data_dir,  ade_checkpoint_path)