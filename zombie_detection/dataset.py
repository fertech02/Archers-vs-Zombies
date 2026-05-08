import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from zombie_detection.cnn import MAX_ZOMBIES

DATASET_FRAME_WH = (320, 180) # size frames were saved at by collect_dataset (WIDTH HEIGHT)
CNN_INPUT_SIZE = (90, 160)   # size the CNN actually expects (HEIGHT WIDTH )


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
