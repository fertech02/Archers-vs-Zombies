import torch
import torch.nn as nn

MAX_ZOMBIES = 8


class ZombieCNN(nn.Module):

    def __init__(self, input_shape: tuple = (3, 90, 160)):
        super().__init__()
        C, H, W = input_shape

        self.backbone = nn.Sequential(
            nn.Conv2d(C,  16, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )

        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, C, H, W))
            self.grid_h, self.grid_w = int(feat.shape[2]), int(feat.shape[3])

        self.detection_head = nn.Conv2d(32, 5, kernel_size=1)

        gy = torch.arange(self.grid_h).view(1, 1, self.grid_h, 1).float()
        gx = torch.arange(self.grid_w).view(1, 1, 1, self.grid_w).float()
        self.register_buffer("_gy", gy, persistent=False)
        self.register_buffer("_gx", gx, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        raw  = self.detection_head(feat)
        B, _, gh, gw = raw.shape

        conf = torch.sigmoid(raw[:, 0:1])
        dx   = torch.sigmoid(raw[:, 1:2])
        dy   = torch.sigmoid(raw[:, 2:3])
        w    = torch.sigmoid(raw[:, 3:4])
        h    = torch.sigmoid(raw[:, 4:5])

        x_global = (self._gx + dx) / gw
        y_global = (self._gy + dy) / gh

        out = torch.cat([conf, x_global, y_global, w, h], dim=1)
        return out.permute(0, 2, 3, 1).reshape(B, gh * gw, 5)
