import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DrivableScore(nn.Module):
    def __init__(
        self,
        num_circles=3,
        ego_width=2.297,
        ego_front_length=4.049,
        ego_rear_length=1.127,
        resolution=0.2,
        eps= 1e-6,
        omega_d=1.0
    ) -> None:
        super().__init__()

        ego_length = ego_front_length + ego_rear_length
        interval = ego_length / num_circles

        self.N = num_circles
        self.width = ego_width
        self.length = ego_length
        self.rear_length = ego_rear_length
        self.resolution = resolution
        self.eps = eps 
        self.radius = math.sqrt(ego_width**2 + interval**2) / 2 - resolution
        self.offset = torch.Tensor(
            [-ego_rear_length + interval / 2 * (2 * i + 1) for i in range(num_circles)]
        )
        self.omega_d = omega_d

    def forward(self, trajectory: Tensor, sdf: Tensor):
        """
        trajectory: (bs, T, 4) - [x, y, cos0, sin0]
        sdf: (bs, H, W), data["cost_maps"][:bs, :, :, 0].float()
        """
        bs, H, W = sdf.shape

        origin_offset = torch.tensor([W // 2, H // 2], device=sdf.device)
        offset = self.offset.to(sdf.device).view(1, 1, self.N, 1)
        # (bs, T, N, 2)
        # (bs, T, 3) -> (bs, T, 2) for [x, y]
        positions = trajectory[..., :2]
        # (bs, T, 3) -> (bs, T) for yaw
        yaw = trajectory[..., 2]
        # Compute cos(yaw) and sin(yaw)
        cos_theta = torch.cos(yaw)       # (bs, T)
        sin_theta = torch.sin(yaw)       # (bs, T)

        # Stack [cos(yaw), sin(yaw)] to create the direction vector
        direction = torch.stack([cos_theta, sin_theta], dim=-1)  # (bs, T, 2)

        # Compute centers using the direction and offset
        centers = positions[..., None, :] + offset * direction[..., None, :]
        # centers = trajectory[..., None, :2] + offset * trajectory[..., None, 2:4]

        pixel_coord = torch.stack(
            [centers[..., 0] / self.resolution, -centers[..., 1] / self.resolution],
            dim=-1,
        )
        grid_xy = pixel_coord / origin_offset
        valid_mask = (grid_xy < 0.95).all(-1) & (grid_xy > -0.95).all(-1)
        on_road_mask = sdf[:, H // 2, W // 2] > 0

        # (bs, T, N)
        distance = F.grid_sample(
            sdf.unsqueeze(1), grid_xy, mode="bilinear", padding_mode="zeros"
        ).squeeze(1)

        cost = self.radius - distance
        valid_mask = valid_mask & (cost > 0) & on_road_mask[:, None, None]
        penalty = torch.relu(cost)
        penalty.masked_fill_(~valid_mask, 0)
        numerator = torch.sum(penalty, dim=(1,2))  # shape=(bs,)
        denominator = torch.sum(valid_mask, dim=(1,2)) + self.eps  # shape=(bs,)
        energy = (1.0 / self.omega_d) * (numerator / denominator)

        return energy