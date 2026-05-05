import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from zombie_detection.cnn import MAX_ZOMBIES

DATASET_FRAME_WH = (320, 180)
CNN_INPUT_SIZE = (90, 160)


class ZombieDataset(Dataset):

    def __init__(
        self,
        frames: np.ndarray,
        labels: list,
        input_size: tuple = CNN_INPUT_SIZE,
        frame_wh: tuple = DATASET_FRAME_WH,
    ):

        self.frames = frames
        self.labels = labels
        self.input_size = input_size
        self.frame_wh = frame_wh

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame = self.frames[idx]
        boxes = self.labels[idx]

        H_out, W_out = self.input_size
        img = Image.fromarray(frame).resize((W_out, H_out), Image.BILINEAR)
        frame_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        orig_W, orig_H = self.frame_wh
        target = np.zeros((MAX_ZOMBIES, 5), dtype=np.float32)
        k = min(len(boxes), MAX_ZOMBIES)
        if k > 0:
            target[:k, 0] = 1.0
            target[:k, 1] = boxes[:k, 0] / orig_W
            target[:k, 2] = boxes[:k, 1] / orig_H
            target[:k, 3] = boxes[:k, 2] / orig_W
            target[:k, 4] = boxes[:k, 3] / orig_H

        return frame_t, torch.from_numpy(target)
