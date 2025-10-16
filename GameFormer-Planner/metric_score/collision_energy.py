import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ESDFCollisionEnergy(nn.Module):
    def __init__(
        self,
        num_circles=3,
        ego_width=2.297,
        ego_front_length=4.049,
        ego_rear_length=1.127,
        resolution=0.2,
    ) -> None:
        super().__init__()

        ego_length = ego_front_length + ego_rear_length
        interval = ego_length / num_circles

        self.N = num_circles
        self.width = ego_width
        self.length = ego_length
        self.rear_length = ego_rear_length
        self.resolution = resolution

        self.radius = math.sqrt(ego_width**2 + interval**2) / 2 - resolution
        self.offset = torch.Tensor(
            [-ego_rear_length + interval / 2 * (2 * i + 1) for i in range(num_circles)]
        )

    def forward(self, trajectory: torch.Tensor, sdf: torch.Tensor):
        """
        trajectory: (bs, T, 4) - [x, y, cosθ, sinθ]
        sdf: (bs, H, W), data["cost_maps"][:bs, :, :, 0].float()
        """
        bs, H, W = sdf.shape

        origin_offset = torch.tensor([W // 2, H // 2], device=sdf.device)
        offset = self.offset.to(sdf.device).view(1, 1, self.N, 1)
 
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
 
        distance = F.grid_sample(
            sdf.unsqueeze(1), grid_xy, mode="bilinear", padding_mode="zeros"
        ).squeeze(1)
 
        cost = self.radius - distance
        cost = torch.relu(cost)        # max(·,0)
        energy = torch.exp(cost) - 1  # Ψ(x)
        energy.masked_fill_(~valid_mask, 0)
        
        return energy.sum(dim=(1,2))  # (B,)
        