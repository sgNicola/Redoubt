import time
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib import cm
# To avoide meshgrid warning
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import multiprocessing as mp
import numpy as np  

os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"
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
        try:
            data = torch.load(self.file_paths[index])
        except Exception as e:
            print(f"Error loading file: {self.file_paths[index]}, Exception: {e}")
            raise e
        # data = torch.load(self.file_paths[index])
        item_data = dict()
        item_data["hidden_feature"] = data["hidden_feature"].clone().detach()
        item_data["ADE"] = data["metrics"]['minADE'].clone().detach() 
        item_data["collision_energy"]= data["collision_energy"].clone().detach() 
        item_data["drivable_score"]= data["drivable_score"].clone().detach()
        return item_data
 

    def __len__(self):
        return len(self.file_paths)
    
def custom_collate_fn(batch):
    
    ordered_batch = {
        "hidden_features": [],
        "ADE": [],
        "collision_energy": [],
        "drivable_score": []
    }
    
    for item in batch:
        hidden_chunks = torch.chunk(item["hidden_feature"], 1, dim=0)
        ade_chunks = torch.chunk(item["ADE"].view(-1), 1, dim=0)
        collision_chunks = torch.chunk(item["collision_energy"], 1, dim=0)
        drivable_chunks = torch.chunk(item["drivable_score"], 1, dim=0)
        
 
        for h, a, c, d in zip(hidden_chunks, ade_chunks, 
                            collision_chunks, drivable_chunks):
            ordered_batch["hidden_features"].append(h.squeeze(0))
            ordered_batch["ADE"].append(a.squeeze(0))
            ordered_batch["collision_energy"].append(c.squeeze(0))
            ordered_batch["drivable_score"].append(d.squeeze(0))
    
    return {
        k: torch.stack(v) for k, v in ordered_batch.items()
    }