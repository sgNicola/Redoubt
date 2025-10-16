import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from torch import Tensor

class DrivableAreaScore(nn.Module):
    def __init__(
        self,
        ego_width: float = 2.297,
        ego_length: float = 4.049 + 1.127,  # Front + rear length
        resolution: float = 0.2,
        violation_threshold: float = 0.5,  # SDF距离阈值（可行驶区域外判定）
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.violation_threshold = violation_threshold
        self.eps = eps
        
        # 定义车辆四个角落相对于中心的偏移（前左FL, 后左RL, 后右RR, 前右FR）
        half_width = ego_width / 2
        half_length = ego_length / 2
        self.corner_offsets = torch.Tensor([
            [half_length, half_width],   # FL
            [-half_length, half_width],  # RL
            [-half_length, -half_width], # RR
            [half_length, -half_width]   # FR
        ])  # Shape: [4, 2]

    def compute_corners(self, trajectory: Tensor) -> Tensor:
        """
        计算轨迹中每个时刻车辆四个角落的坐标
        :param trajectory: (bs, T, 4) [x, y, cosθ, sinθ]
        :return: corners (bs, T, 4, 2)
        """
        bs, T, _ = trajectory.shape
        # 旋转偏移量到车辆朝向
        rotation = torch.stack([
            trajectory[..., 2], -trajectory[..., 3],
            trajectory[..., 3],  trajectory[..., 2]
        ], dim=-1).view(bs, T, 2, 2)  # 旋转矩阵 [bs, T, 2, 2]
        
        # 计算四个角落的偏移 [bs, T, 4, 2]
        offsets = torch.matmul(
            self.corner_offsets.to(trajectory.device)[None, None], 
            rotation
        )
        # 添加中心坐标 [bs, T, 4, 2]
        corners = trajectory[..., None, :2] + offsets
        return corners

    def forward(
        self, 
        trajectory: Tensor, 
        sdf: Tensor, 
        return_details: bool = False
    ) -> Tensor:
        """
        :param trajectory: (bs, T, 4) [x, y, cosθ, sinθ]
        :param sdf: (bs, H, W), SDF值>0表示可行驶区域
        :return: energy (bs,) 越小表示越合规
        """
        bs, H, W = sdf.shape
        corners = self.compute_corners(trajectory)  # (bs, T, 4, 2)
        
        # 将坐标转换为SDF采样网格坐标 [-1,1]
        pixel_coord = torch.stack([
            corners[..., 0] / self.resolution, 
            -corners[..., 1] / self.resolution
        ], dim=-1)  # (bs, T, 4, 2)
        origin_offset = torch.tensor([W//2, H//2], device=sdf.device)
        grid_xy = pixel_coord / origin_offset  # 归一化到 [-1,1]
        
        # 有效性掩码（防止采样超出SDF边界）
        valid_mask = (grid_xy.abs() < 0.95).all(dim=-1)  # (bs, T, 4)
        
        # 双线性插值采样SDF值 (bs, T, 4)
        sdf_values = F.grid_sample(
            sdf.unsqueeze(1), 
            grid_xy.view(bs, -1, 1, 2), 
            mode="bilinear", 
            padding_mode="zeros"
        ).view(bs, -1, T, 4).squeeze(1)  # (bs, T, 4)
        
        # 计算违规惩罚：SDF < threshold 的角落视为违规
        violation_mask = (sdf_values < self.violation_threshold) & valid_mask
        penalty = torch.relu(self.violation_threshold - sdf_values)  # 距离阈值越近惩罚越大
        
        # 统计有效点中的违规比例
        total_violations = violation_mask.sum(dim=(1,2))  # (bs,)
        total_valid = valid_mask.sum(dim=(1,2)) + self.eps  # (bs,)
        energy = total_violations / total_valid
        
        return (energy, (violation_mask, penalty)) if return_details else energy