"""
Preprocessing utilities shared between training and inference.
"""
import numpy as np
import torch
from PIL import Image

CNN_INPUT_SIZE = (90, 160)  # (H, W) expected by ZombieCNN — aspect-preserving 16:9


def preprocess_obs(obs: np.ndarray, input_size: tuple = CNN_INPUT_SIZE) -> torch.Tensor:
    """
    Prepare a raw environment observation for ZombieCNN.

    Args:
        obs       : (H, W, 3) uint8 — raw frame from the environment
        input_size: (H, W) target size for the CNN

    Returns:
        (1, 3, H, W) float32 tensor in [0, 1], ready for model(x)
    """
    H, W = input_size
    img = Image.fromarray(obs).resize((W, H), Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)   # add batch dim


def decode_detections(
    preds: torch.Tensor,
    conf_threshold: float = 0.5,
    orig_w: int = 1280,
    orig_h: int = 720,
) -> np.ndarray:
    """
    Convert ZombieCNN output to bounding boxes in original pixel coordinates.

    Args:
        preds          : (1, MAX_ZOMBIES, 5) tensor from model.forward()
        conf_threshold : minimum confidence to keep a box
        orig_w, orig_h : size of the original (un-preprocessed) frame

    Returns:
        (k, 4) float32 array of [x, y, w, h] in pixel space,
        sorted by confidence descending (most confident first).
        Returns shape (0, 4) when no zombie is detected.
    """
    preds_np = preds[0].detach().cpu().numpy()   # (MAX_ZOMBIES, 5)
    mask     = preds_np[:, 0] >= conf_threshold
    detected = preds_np[mask]

    if len(detected) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    detected = detected[np.argsort(-detected[:, 0])]   # sort by confidence

    boxes = np.stack([
        detected[:, 1] * orig_w,   # x
        detected[:, 2] * orig_h,   # y
        detected[:, 3] * orig_w,   # w
        detected[:, 4] * orig_h,   # h
    ], axis=1).astype(np.float32)

    return boxes
