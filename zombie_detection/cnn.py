"""
Shared CNN backbone for zombie detection and RL feature extraction.

Architecture (YOLO-tiny detection head):
  backbone (Conv + BN layers) → spatial feature map (64, gh, gw)
                                              ├─ 1×1 conv → (5, gh, gw) → reshape (gh·gw, 5)
                                              │             per-cell (conf, x, y, w, h) in [0, 1]
                                              └─ flatten + fc → 512-dim feature vector (RL)

Each grid cell is responsible for predicting a zombie whose top-left corner
falls within that cell. This grounds localization in spatial position rather
than slot index, which is what makes the previous fixed-slot head collapse to
positional priors.
"""
import torch
import torch.nn as nn

MAX_ZOMBIES = 8  # kept for dataset.py target-tensor shape


class ZombieCNN(nn.Module):

    def __init__(self, input_shape: tuple = (3, 90, 160)):
        """
        Args:
            input_shape: (C, H, W) of the preprocessed input frame.
        """
        super().__init__()
        C, H, W = input_shape

        self.backbone = nn.Sequential(
            nn.Conv2d(C,  32, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )

        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, C, H, W))
            self.grid_h, self.grid_w = int(feat.shape[2]), int(feat.shape[3])
            _flat = feat.flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(_flat, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.feat_size = 512

        # YOLO-tiny detection head: small refinement conv + 1×1 to (5, gh, gw)
        self.detection_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 5, kernel_size=1),
        )

        # Cached grid offsets (registered as buffers so they move with .to(device))
        gy = torch.arange(self.grid_h).view(1, 1, self.grid_h, 1).float()
        gx = torch.arange(self.grid_w).view(1, 1, 1, self.grid_w).float()
        self.register_buffer("_gy", gy, persistent=False)
        self.register_buffer("_gx", gx, persistent=False)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the 512-dim feature vector used by the RL agent.
        Input:  (B, C, H, W) float32 in [0, 1]
        Output: (B, 512)
        """
        return self.fc(self.backbone(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-cell zombie detection predictions.
        Input:  (B, C, H, W) float32 in [0, 1]
        Output: (B, gh·gw, 5)
          dim 0 of last axis → confidence in [0, 1]
          dim 1, 2           → (x, y) top-left corner normalized to [0, 1]
          dim 3, 4           → (w, h) normalized to [0, 1]
        """
        feat = self.backbone(x)                       # (B, 64, gh, gw)
        raw  = self.detection_head(feat)              # (B, 5, gh, gw)
        B, _, gh, gw = raw.shape

        conf = torch.sigmoid(raw[:, 0:1])             # (B, 1, gh, gw)
        dx   = torch.sigmoid(raw[:, 1:2])             # offset within cell ∈ [0,1]
        dy   = torch.sigmoid(raw[:, 2:3])
        w    = torch.sigmoid(raw[:, 3:4])             # size ∈ [0,1]
        h    = torch.sigmoid(raw[:, 4:5])

        x_global = (self._gx + dx) / gw               # global top-left x ∈ [0,1]
        y_global = (self._gy + dy) / gh

        out = torch.cat([conf, x_global, y_global, w, h], dim=1)   # (B, 5, gh, gw)
        return out.permute(0, 2, 3, 1).reshape(B, gh * gw, 5)