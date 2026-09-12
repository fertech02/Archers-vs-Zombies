import numpy as np
import torch
from PIL import Image
"""
     (H, W) fed to the CNN. 176 = 44 * 4 and 320 = 80 * 4, so the stride-4 grid
     tiles the frame exactly (see the note in cnn.py). The dataset is stored at
     320x180; labels are normalized against the frame, so the 180 -> 176 resize
     leaves the targets untouched.
"""
CNN_INPUT_SIZE = (176, 320)

DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_NMS_IOU = 0.4

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
        nms serves to remove duplicates when nearby cells see the same zombie.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = boxes[:, 2] * boxes[:, 3]

    order = np.argsort(-scores)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        # Calculate intersection
        # max -> top-left ; min -> bottom-right
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])


        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter

        # intersection over union in the range [0,1]
        iou = np.where(union > 0, inter / union, 0.0)

        # Keeps only the ones under the iou_threshold
        order = rest[iou < iou_threshold]

    return np.array(keep, dtype=np.int64)


def preprocess_obs(observation: np.ndarray, input_size: tuple = CNN_INPUT_SIZE) -> torch.Tensor:

    H_out, W_out = input_size

    if observation.dtype != np.uint8:
        observation = observation.astype(np.uint8)

    if observation.shape[:2] != (H_out, W_out):
        observation = np.array(
            Image.fromarray(observation).resize((W_out, H_out), Image.BILINEAR)
        )

    tensor = torch.from_numpy(observation).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)

def decode_detections(
    preds: torch.Tensor,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_NMS_IOU,
    orig_w: int = 1280,
    orig_h: int = 720,
) -> np.ndarray:

    """
        From CNN output to relevant detections.
    """

    # Takes first image of batch and put it into CPU
    preds_np = preds[0].detach().cpu().numpy()
    # Keeps only boxes over the conf_threshold
    mask     = preds_np[:, 0] >= conf_threshold
    detected = preds_np[mask]

    if len(detected) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    w = detected[:, 3] * orig_w
    h = detected[:, 4] * orig_h
    boxes = np.stack([
        detected[:, 1] * orig_w - w / 2.0,   # centre -> left edge
        detected[:, 2] * orig_h - h / 2.0,   # centre -> top edge
        w,
        h,
    ], axis=1).astype(np.float32)

    # Applies nms to the boxes and keeps only the ones below the iou_threshold
    keep = nms(boxes, detected[:, 0], iou_threshold)
    return boxes[keep]


def load_detector_config(path: str = None) -> dict:
    """
    Thresholds picked on the validation split by train.py. Falls back to the
    module defaults when the file has not been produced yet.
    """
    import json
    import os

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "detector_config.json")

    cfg = {
        "conf_threshold_f1": DEFAULT_CONF_THRESHOLD,
        "conf_threshold_recall": DEFAULT_CONF_THRESHOLD,
        "nms_iou": DEFAULT_NMS_IOU,
    }
    try:
        with open(path) as fh:
            cfg.update(json.load(fh))
    except (OSError, ValueError):
        pass
    return cfg
