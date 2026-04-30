"""
Shared CNN backbone for zombie detection and RL feature extraction.

Architecture:
  backbone (Conv layers) → fc → 512-dim feature vector
                                       ↓
                            detection_head → (MAX_ZOMBIES, 5)
                                             (confidence, x, y, w, h) normalized [0,1]

The 512-dim feature vector is also used by the RLlib policy/value heads.
"""
import torch
import torch.nn as nn

MAX_ZOMBIES = 8


class ZombieCNN(nn.Module):

    def __init__(self, input_shape: tuple = (3, 84, 84)):
        """
        Args:
            input_shape: (C, H, W) of the preprocessed input frame.
        """
        super().__init__()
        C, H, W = input_shape

        self.backbone = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            _flat = self.backbone(torch.zeros(1, C, H, W)).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(_flat, 512),
            nn.ReLU(),
        )
        self.feat_size = 512

        # Each zombie slot: (confidence, x, y, w, h) all normalized to [0,1]
        self.detection_head = nn.Linear(512, MAX_ZOMBIES * 5)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the 512-dim feature vector used by the RL agent.
        Input:  (B, C, H, W) float32 in [0, 1]
        Output: (B, 512)
        """
        return self.fc(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns zombie detection predictions.
        Input:  (B, C, H, W) float32 in [0, 1]
        Output: (B, MAX_ZOMBIES, 5)
          dim 0 of last axis → confidence in [0, 1]
          dim 1-4            → (x, y, w, h) normalized to [0, 1]
        """
        feats = self.extract_features(x)
        raw = self.detection_head(feats).view(-1, MAX_ZOMBIES, 5)
        conf = torch.sigmoid(raw[:, :, 0:1])
        bbox = torch.sigmoid(raw[:, :, 1:])
        return torch.cat([conf, bbox], dim=-1)
