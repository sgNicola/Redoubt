import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from src.metric_score.min_ade import minADE


logger = logging.getLogger(__name__)

class ComputeMetric():
    def __init__(
        self,
        device =None
    ) -> None:
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history_steps = 21 #model.histroy_steps
        self.metrics  = MetricCollection(
        [
            minADE().to(self.device),
        ]
    )

    def _compute_metrics(self, res, data) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        # get top 6 modes
        trajectory, probability = res["trajectory"], res["probability"]
        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]
        metrics = self.metrics(outputs, target)
        return metrics
