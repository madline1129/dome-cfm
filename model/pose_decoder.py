import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class PoseDecoder(BaseModule):
    """OccWorld-style MLP decoder for multi-mode future ego poses."""

    def __init__(
        self,
        in_channels,
        num_layers=2,
        num_modes=3,
        num_fut_ts=6,
        out_dim=2,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.num_modes = int(num_modes)
        self.num_fut_ts = int(num_fut_ts)
        self.out_dim = int(out_dim)

        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_channels, in_channels),
                nn.ReLU(True),
            ])
        layers.append(nn.Linear(in_channels, self.num_modes * self.num_fut_ts * self.out_dim))
        self.pose_decoder = nn.Sequential(*layers)

    def forward(self, x):
        pose = self.pose_decoder(x)
        return pose.reshape(*x.shape[:-1], self.num_modes, self.num_fut_ts, self.out_dim)
