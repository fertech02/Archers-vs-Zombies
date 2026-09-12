import math

import torch
import torch.nn as nn


MAX_ZOMBIES = 8

# Zombie sprites are a fixed 29x31 px box in the 1280x720 render, so they always
# cover this fraction of the frame whatever the input resolution.
ANCHOR_W = 29.0 / 1280.0
ANCHOR_H = 31.0 / 720.0

# Only ~0.06% of the cells hold a zombie, so the confidence output is initialised
# low.
PRIOR_CONF = 0.01


class ZombieCNN(nn.Module):
    """
    Single-class zombie detector with a YOLO-style head.
    The image is divided in a grid (grid_h, grid_w) of cells. Each cell has to
    predict a box if the centre of a zombie falls inside it.

    Cell offsets (dx, dy) are added to the cell's (gx, gy) index and normalized
    by the grid size to obtain absolute box centres in [0, 1]. Width and height
    are not predicted: the sprite size is fixed, so they are emitted as the
    constant anchor.

    Offsets use the YOLOv5 form sigmoid(t) * 2 - 0.5, i.e. a range of
    [-0.5, 1.5] cells. A cell may therefore claim a centre that sits slightly
    outside its own bounds, which is what lets the responsible cell (the one
    holding the centre) still reach a box whose centre lands right on a border,
    where a plain sigmoid would need an infinite logit.

    Output shape: (B, grid_h * grid_w, 5) where each row is
    (confidence, x_centre, y_centre, width, height), all normalized to [0, 1] of
    the full frame.
    """

    def __init__(self, input_shape: tuple = (3, 176, 320)):
        super().__init__()
        C, H, W = input_shape

        self.backbone = nn.Sequential(
            nn.Conv2d(C,  16, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )

        with torch.no_grad():
            # Forward to measure grid_h, grid_w
            feat = self.backbone(torch.zeros(1, C, H, W))
            self.grid_h, self.grid_w = int(feat.shape[2]), int(feat.shape[3])

        self.detection_head = nn.Sequential(
            # Reduce the 32 channels to 3: (conf_logit, dx, dy)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1),
        )

        # Bias the last layer on the priors, so at step 0 the model already
        # predicts "no zombie, center of the cell".
        out_conv = self.detection_head[-1]
        nn.init.normal_(out_conv.weight, std=0.01)
        with torch.no_grad():
            logit = lambda v: math.log(v / (1.0 - v))
            out_conv.bias[0] = logit(PRIOR_CONF)
            out_conv.bias[1] = 0.0   # dx -> sigmoid(0) * 2 - 0.5 = middle of the cell
            out_conv.bias[2] = 0.0   # dy

        # Row/Column index of each cell
        gy = torch.arange(self.grid_h).view(1, 1, self.grid_h, 1).float()
        gx = torch.arange(self.grid_w).view(1, 1, 1, self.grid_w).float()

        # Register as part of the module, automatically moved to GPU with cuda.
        # To not train, not saved on state_dict
        self.register_buffer("_gy", gy, persistent=False)
        self.register_buffer("_gx", gx, persistent=False)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        feat = self.backbone(x)
        raw  = self.detection_head(feat)
        B, _, gh, gw = raw.shape

        # Raw confidence logit. Kept unsquashed for the training loss, which
        # uses binary_cross_entropy_with_logits.
        conf_logit = raw[:, 0:1]
        conf = torch.sigmoid(conf_logit)

        # Offsets within the cell, in [-0.5, 1.5]
        dx = torch.sigmoid(raw[:, 1:2]) * 2.0 - 0.5
        dy = torch.sigmoid(raw[:, 2:3]) * 2.0 - 0.5

        # Grid position + offset divided by grid dimension to obtain
        # normalized coordinates with respect to the full image.
        x_centre = (self._gx + dx) / gw
        y_centre = (self._gy + dy) / gh

        # Constant sprite size, broadcast to every cell.
        w = torch.full_like(x_centre, ANCHOR_W)
        h = torch.full_like(y_centre, ANCHOR_H)

        out = torch.cat([conf, x_centre, y_centre, w, h], dim=1)
        out = out.permute(0, 2, 3, 1).reshape(B, gh * gw, 5)

        if return_logits:
            return out, conf_logit.reshape(B, gh * gw)
        return out
