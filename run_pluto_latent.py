import logging
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker 
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from src.metric_score.collision_energy import ESDFCollisionEnergy
from src.metric_score.drivable_area_energy import DrivableScore
from src.metric_score.ade_score import ComputeMetric
from src.models.pluto.loss.esdf_collision_loss import ESDFCollisionLoss
from src.models.pluto.inference_pluto import PlanningModel
import os
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
from src.custom_training import (
    build_training_engine,
    update_config_for_training,
)

CONFIG_PATH = "./config"
CONFIG_NAME = "default_training"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)
    worker = build_worker(cfg)
    update_config_for_training(cfg)
    
    with ProfilerContextManager(
        cfg.output_dir, cfg.enable_profiling, "build_training_engine"
    ):
        engine = build_training_engine(cfg, worker)
    
    datamodule=engine.datamodule
    
    if cfg.stage == 'train':
        datamodule.setup(stage= "fit") 
        dataloader = datamodule.train_dataloader()
    elif cfg.stage == 'val':
        datamodule.setup(stage= "validate") 
        dataloader = datamodule.val_dataloader()
        
    model = engine.model
    inference_model = PlanningModel()
    device = 'cpu'
    pluto_dataset = os.getenv("PLUTO")   
    OT =80
    radius = model.radius
    num_modes = model.num_modes
    mode_interval = radius / num_modes
    esdfcollision_loss = ESDFCollisionLoss()
    esdfcollision_energy = ESDFCollisionEnergy()
    
    drivable = DrivableScore()
    if cfg.checkpoint is not None:
        ckpt = torch.load(cfg.checkpoint, map_location=torch.device("cpu"))
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        inference_model.load_state_dict(state_dict)
        inference_model = inference_model.to(device)
        inference_model.eval()
    metric_computer=ComputeMetric()
    save_dir = f"{pluto_dataset}/{cfg.stage}_results"
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            features, targets, scenarios = batch
            res, hidden_feature= inference_model.forward(features["feature"].data)
            trajectory = res["trajectory"]
            bs, R, M, T, _ =  res["trajectory"].shape
            end = -OT+T if T < OT else None
            data =features["feature"].data
            valid_mask = features["feature"].data["agent"]["valid_mask"][:bs, :, OT:end]
            valid_mask=valid_mask[:, 0]
            num_valid_points = valid_mask.sum(-1)
            endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
            r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)  # (bs, R)
            future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index]
            
            target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
            )
            target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / mode_interval
            ).long()
            target_m_index.clamp_(min=0, max=num_modes - 1)
            metrics = metric_computer._compute_metrics(res, features["feature"].data)
            best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index]
            collision_loss= esdfcollision_loss(best_trajectory, data["cost_maps"][:bs, :, :, 0].float())
            collision_energy= esdfcollision_energy(best_trajectory, data["cost_maps"][:bs, :, :, 0].float())
            drivable_score =  drivable(best_trajectory, features["feature"].data["cost_maps"][:bs, :, :, 0].float())
            results = dict()
            results["res"] = res
            results["hidden_feature"] = hidden_feature
            results["metrics"] = metrics
            results["collision_energy"] = collision_energy
            results["drivable_score"] = drivable_score
            results["collision_loss"] = collision_loss
            save_path = f"{pluto_dataset}/{cfg.stage}_results/{batch_index}.pt"
            torch.save(
                results, save_path
            )
          
if __name__ == "__main__":
    main()