"""
PyTorch Dataset for zombie detection training.

Each sample is a (frame_tensor, target_tensor) pair:
  frame_tensor : (3, H, W) float32 in [0, 1]  — resized to CNN_INPUT_SIZE
  target_tensor: (MAX_ZOMBIES, 5) float32
                  col 0 → confidence (1 if real zombie, 0 if padding)
                  col 1 → x  normalized to [0, 1] over original frame width
                  col 2 → y  normalized to [0, 1] over original frame height
                  col 3 → w  normalized to [0, 1] over original frame width
                  col 4 → h  normalized to [0, 1] over original frame height

Ground-truth boxes beyond MAX_ZOMBIES are discarded (rare in KAZ).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from zombie_detection.cnn import MAX_ZOMBIES

# Saved frames are at this resolution (W, H) — see collect_dataset.py
DATASET_FRAME_WH = (320, 180)

# CNN input size (H, W)
CNN_INPUT_SIZE = (90, 160)


class ZombieDataset(Dataset):

    def __init__(
        self,
        frames: np.ndarray,
        labels: list,
        input_size: tuple = CNN_INPUT_SIZE,
        frame_wh: tuple = DATASET_FRAME_WH,
    ):
        """
        Args:
            frames   : (N, H, W, 3) uint8 array from load_dataset()
            labels   : list of N arrays, each (k, 4) float32 [x, y, w, h]
            input_size: (H, W) to resize frames for CNN
            frame_wh : (W, H) of the saved frames (for bbox normalization)
        """
        self.frames = frames
        self.labels = labels
        self.input_size = input_size      # (H, W)
        self.frame_wh = frame_wh          # (W, H)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame = self.frames[idx]          # (H, W, 3) uint8
        boxes = self.labels[idx]          # (k, 4) float32 [x, y, w, h]

        # --- resize frame ---
        H_out, W_out = self.input_size
        img = Image.fromarray(frame).resize((W_out, H_out), Image.BILINEAR)
        frame_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # --- build target tensor ---
        orig_W, orig_H = self.frame_wh
        target = np.zeros((MAX_ZOMBIES, 5), dtype=np.float32)
        k = min(len(boxes), MAX_ZOMBIES)
        if k > 0:
            target[:k, 0] = 1.0                       # confidence = 1
            target[:k, 1] = boxes[:k, 0] / orig_W    # x normalized
            target[:k, 2] = boxes[:k, 1] / orig_H    # y normalized
            target[:k, 3] = boxes[:k, 2] / orig_W    # w normalized
            target[:k, 4] = boxes[:k, 3] / orig_H    # h normalized

        return frame_t, torch.from_numpy(target)
