from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric

from src.metrics.utils import sort_predictions

class minADE(Metric):
    """Minimum Average Displacement Error
    minADE: The average L2 distance between the best forecasted trajectory and the ground truth.
            The best here refers to the trajectory that has the minimum endpoint error.
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k=6,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(minADE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.k = k
        self.add_state("min_ades", default=[], dist_reduce_fx="cat")

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred, _ = sort_predictions(
                outputs["trajectory"], outputs["probability"], k=self.k
            )
            pred = pred.to(self.device)
            target = target.to(self.device)
            
            ade = torch.norm(
                pred[..., :2] - target.unsqueeze(1)[..., :pred.shape[-2], :2], 
                p=2, 
                dim=-1
            ).mean(-1)
            min_ade = ade.min(-1)[0]
            self.min_ades.append(min_ade)
 
    def compute(self) -> torch.Tensor:
        all_min_ades = torch.cat(self.min_ades, dim=0) 
        return all_min_ades.view(-1).cpu()