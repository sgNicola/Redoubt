import os
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from GameFormer.inference import GameFormer
from torch.utils.data import DataLoader
from GameFormer.train_utils import *
from metric_score.drivable_area_energy import DrivableScore
from metric_score.collision_energy import ESDFCollisionEnergy

def inference_epoch(data_loader, model,stage):
    model.eval()
    esdfcollision_energy = ESDFCollisionEnergy()
    
    drivable = DrivableScore()
    gameformer_dataset = os.getenv("GAMEFORMER")
    save_dir = f"{gameformer_dataset}/{stage}_results"
    os.makedirs(save_dir, exist_ok=True)
    with tqdm(data_loader, desc="Inference", unit="batch") as data_epoch:
        batch_idx = 0
        for batch in data_epoch:
           # prepare data
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_future = batch[5].to(args.device)
            neighbors_future = batch[6].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)
            cost_map = batch[7].to(args.device)
            # call the model
            with torch.no_grad():
                level_k_outputs, ego_plan, env_route_encoding = model(inputs)
                loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
                prediction = results[:, 1:]
                bs, T, _ = ego_plan.shape
                metrics = motion_metrics(ego_plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
                collision_energy= esdfcollision_energy(ego_plan, cost_map[:bs, :, :, 0].float())
                drivable_score =  drivable(ego_plan, cost_map[:bs, :, :, 0].float())
                results = dict()
                results["metrics"] = {}
                results["res"] = ego_plan.detach().cpu()
 
                results["hidden_feature"] = env_route_encoding.detach().cpu()
                results["collision_energy"] = collision_energy.detach().cpu()
                results["drivable_score"] = drivable_score.detach().cpu()
                results["metrics"]["minADE"] = torch.tensor(metrics[0])
                save_path = f"{gameformer_dataset}/{stage}_results/{batch_idx}.pt"
                torch.save(
                results, save_path
                )
            batch_idx += 1

def model_inference():
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Use device: {}".format(args.device))
    # set seed
    set_seed(args.seed)

    # set up model
    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels, neighbors=args.num_neighbors)
    # training parameters
    batch_size = args.batch_size
    gameformer.load_state_dict(torch.load(args.model_path, map_location=args.device))
    gameformer.to(args.device)
    gameformer.eval()
    # set up data loaders
    train_set = DrivingData(args.train_set + '/*.npz', args.num_neighbors)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    stage = args.stage
    inference_epoch(train_loader, gameformer, stage)
       

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
 
    parser.add_argument('--train_set', type=str, help='path to train data')
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--encoder_layers', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_levels', type=int, help='levels of reasoning', default=2)
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=1)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--model_path', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--stage', type=str, help='train or val', default='train')
    args = parser.parse_args()

    # Run
    model_inference()