import torch
import torch.nn as nn

MAX_ZOMBIES = 8

class ZombieCNN(nn.Module):
    """
    Single-class zombie detector with a YOLO-style head.
    The image is divided in a grid (grid_h, grid_w) of cells. Each cell
    has to predict a box if the center of a zombie falls inside it.

    Cell offsets (dx, dy) are added to the cell's (gx, gy) index and normalized
    by the grid size to obtain absolute box centers in [0, 1]. Width and height
    are predicted directly in normalized image coordinates.

    Output shape: (B, grid_h * grid_w, 5) where each row is
    (confidence, x_center, y_center, width, height)
    """

    def __init__(self, input_shape: tuple = (3, 90, 160)):
        super().__init__()
        C, H, W = input_shape

        # Total downsampling: x8
        self.backbone = nn.Sequential(
            nn.Conv2d(C,  16, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )

        with torch.no_grad():
            # Forward to measure grid_h, grid_w
            feat = self.backbone(torch.zeros(1, C, H, W))
            self.grid_h, self.grid_w = int(feat.shape[2]), int(feat.shape[3])

        self.detection_head = nn.Sequential(
            # Reduce the 32 channels to 5: (conf, dx, dy, w, h)
            # Final convolution acts as a classifier per-pixel on the grid
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 5, kernel_size=1),
        )

        # Row/Column index of each cell
        gy = torch.arange(self.grid_h).view(1, 1, self.grid_h, 1).float()
        gx = torch.arange(self.grid_w).view(1, 1, 1, self.grid_w).float()

        # Register as part of the module, automatically moved to GPU with cuda.
        # To not train, not saved on state_dict
        self.register_buffer("_gy", gy, persistent=False)
        self.register_buffer("_gx", gx, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        raw  = self.detection_head(feat)
        B, _, gh, gw = raw.shape

        # Confidence we have the zombie within the cell
        conf = torch.sigmoid(raw[:, 0:1])
        # Sigmoid force the offset to be within the cell
        dx   = torch.sigmoid(raw[:, 1:2])
        dy   = torch.sigmoid(raw[:, 2:3])
        # Normalized width, height
        w    = torch.sigmoid(raw[:, 3:4])
        h    = torch.sigmoid(raw[:, 4:5])

        # Grid position + offset divided by grid dimension to obtain
        # normalized coordinates with respect to the full image.
        x_global = (self._gx + dx) / gw
        y_global = (self._gy + dy) / gh

        out = torch.cat([conf, x_global, y_global, w, h], dim=1)
        return out.permute(0, 2, 3, 1).reshape(B, gh * gw, 5)
