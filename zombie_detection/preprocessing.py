import numpy as np
import torch
from PIL import Image

CNN_INPUT_SIZE = (90, 160)

def preprocess_obs(obs: np.ndarray, input_size: tuple = CNN_INPUT_SIZE) -> torch.Tensor:

    H, W = input_size
    # Resize image into the cnn size and applies bilinear filter (weighted average foreach pixel of
    # the nearby 2x2 pixels)
    img = Image.fromarray(obs).resize((W, H), Image.BILINEAR)
    # Transform it into a torch tensor, restore (C, H, W), normalize it
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
        nms serves to remove duplicates when nearby cells see the same zombie
        and generates almost identical boxes.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    # (x,y,w,h) -> (x1,y1,x2,y2)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    # Box Area
    areas = boxes[:, 2] * boxes[:, 3]
    # Order the boxes by decreasing confidence
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
        # Shared rectangle between box i and the remaining ones
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        # intersection over union in thre range [0,1]
        iou = np.where(union > 0, inter / union, 0.0)
        # Keeps only the ones under the iou_threshold
        order = rest[iou < iou_threshold]

    return np.array(keep, dtype=np.int64)


def decode_detections(
    preds: torch.Tensor,
    conf_threshold: float = 0.6,
    iou_threshold: float = 0.4,
    orig_w: int = 1280,
    orig_h: int = 720,
) -> np.ndarray:

    """
        Transform CNN output into bounding boxes to be used in the game.
    """

    # Takes first image of bash and put it into CPU
    preds_np = preds[0].detach().cpu().numpy()
    # Keeps only boxes over the conf_threshold
    mask     = preds_np[:, 0] >= conf_threshold
    detected = preds_np[mask]

    if len(detected) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.stack([
        detected[:, 1] * orig_w,
        detected[:, 2] * orig_h,
        detected[:, 3] * orig_w,
        detected[:, 4] * orig_h,
    ], axis=1).astype(np.float32)

    # Applies nms to the boxes and keeps only the ones below the iou_threshold
    keep = _nms(boxes, detected[:, 0], iou_threshold)
    return boxes[keep]

