from typing import Any, Callable, Optional, Dict

import torch
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy


class PredAvgFDE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(PredAvgFDE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> None:
        """
        outputs: [A, T, 2]
        target: [A, T, 2]
        """
        with torch.no_grad():
            prediction, valid_mask = outputs["prediction"], outputs["valid_mask"]
            target = outputs["prediction_target"]
            endpoint_mask = valid_mask[..., prediction.shape[-2]-1].float()
            fde = (
                torch.norm(prediction[..., -1, :2] - target[..., prediction.shape[-2]-1, :2], p=2, dim=-1)
                * endpoint_mask
            ).sum(-1)

            self.sum += fde.sum()
            self.count += endpoint_mask.sum().long()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
