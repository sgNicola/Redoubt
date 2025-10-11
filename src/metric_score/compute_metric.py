import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metric_score.min_ade import minADE
from src.optim.warmup_cos_lr import WarmupCosLR

logger = logging.getLogger(__name__)


class ComputeMetric():
    def __init__(
        self,
        device = None
    ) -> None:
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = MetricCollection(
            [
 
                minADE().to(self.device),
            ]
        )

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str, batch_idx:int
    ) -> torch.Tensor:
        features, _, _ = batch
        res = self.forward(features["feature"].data)
        ## may get more than res in here
        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        return losses, metrics

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        trajectory, probability, prediction = (
            res["trajectory"],
            res["probability"],
            res["prediction"],
        )
        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]

        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:]

        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target)
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        agent_reg_loss = F.smooth_l1_loss(
            prediction[agent_mask], agent_target[agent_mask][:, :2]
        )

        loss = ego_reg_loss + ego_cls_loss + agent_reg_loss

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss,
            "cls_loss": ego_cls_loss,
            "prediction_loss": agent_reg_loss,
        }

    def _compute_metrics(self, output, data) -> Dict[str, torch.Tensor]:
        metrics = self.metrics(output, data["agent"]["target"][:, 0])
        return metrics