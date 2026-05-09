## Full explanation — Dataset and Preprocessing from scratch

---

## First — understand what a frame IS physically

The game screen is `1280` pixels wide and `720` pixels tall. Every pixel has 3 colour values: Red, Green, Blue, each from 0 to 255.

Numpy stores this as a 3D array with shape `(720, 1280, 3)` — **height first, then width, then channels**. Always remember: numpy = height first.

```python
frame.shape  # (720, 1280, 3)
#               H     W    C
#               │     │    │
#               │     │    └── 3 colours: [R, G, B]
#               │     └─────── 1280 columns (left to right)
#               └───────────── 720 rows (top to bottom)
```

So to access one pixel:

```python
frame[row, col, channel]

frame[0,    0,   :]  # top-left corner     → e.g. [0, 0, 0]      pure black background
frame[0,    640, :]  # top-center          → e.g. [0, 0, 0]      still background
frame[360,  640, :]  # exact center screen → e.g. [34, 139, 34]  greenish = zombie maybe
frame[719,  0,   :]  # bottom-left corner  → e.g. [80, 80, 80]   grey floor
frame[200,  400, 0]  # one number: RED value of pixel at row 200, col 400 → e.g. 34
frame[200,  400, 1]  # GREEN value of same pixel → e.g. 139
frame[200,  400, 2]  # BLUE value of same pixel  → e.g. 34
```

A zombie sprite is green, so wherever a zombie is drawn you'd see high green, low red, low blue:

```python
frame[200, 400, :]  # [34, 139, 34]  ← greenish = zombie pixel
frame[100, 100, :]  # [0,    0,   0] ← black = background
frame[700, 300, :]  # [80,  80,  80] ← grey = floor
```

---

## The size shrinking — why and how many times

The frame gets shrunk **twice** on its journey to the CNN:

```
Game renders at:    (720,  1280, 3)   H×W×C  native resolution
        │
        │  collect_dataset.py saves at:
        ▼
Saved to disk:      (180,  320,  3)   H×W×C  4x smaller each side = 16x fewer pixels
        │
        │  ZombieDataset resizes to:
        ▼
CNN input:          (90,   160,  3)   H×W×C  half again = 4x fewer pixels than saved
```

Why shrink twice? Disk space and speed. Saving at `1280×720` would use 16x more disk. Then the CNN at `160×90` trains much faster than at `320×180` — and zombies are still clearly visible at that resolution.

**PIL convention warning** — when you call `Image.resize()` it takes `(width, height)` — the opposite of numpy:

```python
# numpy shape:      (H,   W  )
# same image is:    (180, 320)

# PIL resize takes: (W,   H  )
Image.resize(         (320, 180))  # ← PIL, width first
Image.resize(         (160, 90))   # ← to get CNN input size
```

This is confusing but just remember: PIL = width first, numpy = height first.

---

## What `self.labels` actually is — with real numbers

A label is the answer to: **"where exactly is each zombie in this frame?"**

Each zombie is described by 4 numbers: `[x, y, width, height]`

```
x      = how many pixels from the LEFT edge to the zombie's left side
y      = how many pixels from the TOP edge to the zombie's top side
width  = how many pixels wide the zombie is
height = how many pixels tall the zombie is
```

Concrete example in the saved `(320, 180)` frame space:

```python
self.labels[0]  # frame 0 has 2 zombies

# array([[80.,  40.,  20.,  25.],   ← zombie 1
#        [200., 30.,  20.,  25.]])  ← zombie 2
#          │     │     │     │
#          x     y     w     h   (all in pixels of the 320×180 frame)

# zombie 1: top-left corner at pixel (80, 40)
#           20 pixels wide, 25 pixels tall
#           so it occupies columns 80→100, rows 40→65

# zombie 2: top-left corner at pixel (200, 30)
#           same size
#           so it occupies columns 200→220, rows 30→55
```

If a frame has no zombies:

```python
self.labels[5]  # frame 5, no zombies visible
# array([], shape=(0, 4))   ← empty, but still shape (N, 4)
```

---

## Now the 4 attributes of `ZombieDataset`

```python
def __init__(self, frames, labels, input_size, frame_wh):
    self.frames     = frames
    self.labels     = labels
    self.input_size = input_size
    self.frame_wh   = frame_wh
```

**`self.frames`**

```python
self.frames         # shape (5000, 180, 320, 3)
#                              │    │    │   │
#                              │    │    │   └── 3 colours RGB
#                              │    │    └─────── 320 pixels wide
#                              │    └──────────── 180 pixels tall
#                              └───────────────── 5000 screenshots collected

self.frames[0]      # shape (180, 320, 3) — screenshot number 0
self.frames[0][45]  # shape (320, 3)      — row 45 of screenshot 0, all 320 pixels
self.frames[0][45][80]   # shape (3,)     — pixel at row 45, col 80 → e.g. [34, 139, 34]
self.frames[0][45][80][1]  # one number   — green channel of that pixel → e.g. 139
```

**`self.labels`**

```python
self.labels         # Python list of length 5000, one entry per frame

self.labels[0]      # frame 0 had 2 zombies
# array([[80.,  40., 20., 25.],
#        [200., 30., 20., 25.]])   shape (2, 4)

self.labels[3]      # frame 3 had 0 zombies
# array([])   shape (0, 4)

self.labels[7]      # frame 7 had 4 zombies
# array([[45.,  20., 20., 25.],
#        [120., 15., 20., 25.],
#        [210., 35., 20., 25.],
#        [290., 10., 20., 25.]])   shape (4, 4)

# all coordinates are in the (320×180) pixel space of the saved frames
```

**`self.input_size = (90, 160)`**

```python
self.input_size     # (H, W) = (90, 160) — numpy convention, height first
                    # this is the size the CNN expects as input
                    # not the size the frames are stored at
                    # used in __getitem__ to resize before feeding CNN
```

**`self.frame_wh = (320, 180)`**

```python
self.frame_wh       # (W, H) = (320, 180) — PIL convention, width first
                    # this is the size the frames are currently stored at
                    # used to NORMALIZE box coordinates to [0, 1]
                    # x_norm = x_pixels / 320
                    # y_norm = y_pixels / 180
```

---

## What `__getitem__` does step by step

This is called by PyTorch every time it needs one training example. Let's trace it for frame 0 which has 2 zombies:

```python
def __getitem__(self, idx):                    # idx = 0
    frame = self.frames[0]                     # (180, 320, 3) uint8
    boxes = self.labels[0]                     # [[80, 40, 20, 25], [200, 30, 20, 25]]
    
    H_out, W_out = self.input_size             # H_out=90, W_out=160
    
    #Three things happening here in one line:

    #Resize from (320, 180) down to (160, 90) — the CNN input size.
    
    #permute(2, 0, 1) — reorder from (H, W, 3) to (3, H, W) because PyTorch convolutions expect channels first

    #/ 255.0 — normalize from integers [0, 255] to floats [0.0, 1.0]
    img = Image.fromarray(frame)               
          .resize((W_out, H_out), BILINEAR)    
                                               
    
    frame_t = torch.from_numpy(np.array(img))  
              .permute(2, 0, 1)                
              .float() / 255.0                 
    
    # frame_t shape: (3, 90, 160)
    # frame_t[0]  = all red values,   shape (90, 160), values like 0.133
    # frame_t[1]  = all green values, shape (90, 160), values like 0.545
    # frame_t[2]  = all blue values,  shape (90, 160), values like 0.133
```

Now the target — converting variable-length boxes to fixed-size tensor:

Why MAX_ZOMBIES=8 and fixed size?

This is a fundamental PyTorch constraint. When DataLoader batches examples together it stacks tensors — so every example must have the same shape. But different frames have different numbers of zombies — one frame might have 2, another might have 6. You can't stack tensors of different sizes.

Solution: always allocate space for 8 zombies. If a frame has 3 zombies, fill the first 3 slots with real data and leave the last 5 as zeros with confidence=0. The zeros are "empty slots" — they don't represent anything, they're just padding.
So a target looks like this for a frame with 2 zombies:

```
slot 0: [1.0, 0.23, 0.45, 0.06, 0.08]  ← real zombie
slot 1: [1.0, 0.67, 0.31, 0.06, 0.08]  ← real zombie  
slot 2: [0.0, 0.0,  0.0,  0.0,  0.0 ]  ← empty padding
slot 3: [0.0, 0.0,  0.0,  0.0,  0.0 ]  ← empty padding
...
slot 7: [0.0, 0.0,  0.0,  0.0,  0.0 ]  ← empty padding
```
The 5 numbers per slot are [confidence, x, y, width, height] — all normalized to [0, 1] by dividing by frame dimensions.

x,y : position
----- w,h : width height of bounding boxes

Why normalize coordinates? Because the CNN works at (90, 160) but the labels came from (320, 180). Instead of converting between pixel spaces you normalize everything to [0, 1] which is resolution-independent.

```python
    orig_W, orig_H = self.frame_wh             # orig_W=320, orig_H=180

    target = np.zeros((8, 5), dtype=np.float32)
    # starts as all zeros:
    # [[0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    # ENCORE 6 AUTRES, 8 AU TOTAL]

    k = min(len(boxes), 8)                     # k = min(2, 8) = 2

    #normalize BETWEEN 0 AND 1 WHICH IS REOLUTION INDEPENDANT
    target[:2, 0] = 1.0                        # confidence = 1 for real zombies
    target[:2, 1] = boxes[:2, 0] / orig_W      # x: [80/320,  200/320] = [0.25,  0.625]
    target[:2, 2] = boxes[:2, 1] / orig_H      # y: [40/180,  30/180]  = [0.222, 0.167]
    target[:2, 3] = boxes[:2, 2] / orig_W      # w: [20/320,  20/320]  = [0.0625, 0.0625]
    target[:2, 4] = boxes[:2, 3] / orig_H      # h: [25/180,  25/180]  = [0.139, 0.139]

    # target is now:
    # [[1.0,  0.25,   0.222,  0.0625, 0.139],  ← zombie 1, real
    #  [1.0,  0.625,  0.167,  0.0625, 0.139],  ← zombie 2, real
    #  [0.0,  0.0,    0.0,    0.0,    0.0  ],  ← empty slot
    #  [0.0,  0.0,    0.0,    0.0,    0.0  ],  ← empty slot
    #  [0.0,  0.0,    0.0,    0.0,    0.0  ],  ← empty slot
    #  [0.0,  0.0,    0.0,    0.0,    0.0  ],  ← empty slot
    #  [0.0,  0.0,    0.0,    0.0,    0.0  ],  ← empty slot
    #  [0.0,  0.0,    0.0,    0.0,    0.0  ]]  ← empty slot

    return frame_t, torch.from_numpy(target)
    # frame_t: (3, 90, 160) float tensor — the image
    # target:  (8, 5)       float tensor — the zombie locations
```

---

## `preprocessing.py` — same idea but for live gameplay

During training you had time to load from disk, resize carefully, build targets. During gameplay you have milliseconds. So `preprocessing.py` does the minimum needed to prepare a frame for the CNN:

It does two things:
1. prepare a frame for the CNN during gameplay
2. and decode the CNN's output back into usable bounding boxes.


```python
def preprocess_obs(obs, input_size=(90, 160)):
"""Almost identical to what dataset.py does — resize, permute, normalize. The one difference is .unsqueeze(0) which adds a batch dimension. The CNN expects (batch, channels, H, W) — during training batch is 64, but during gameplay you're processing one frame at a time so you fake a batch of 1: (1, 3, 90, 160)."""

    # obs comes from env.observe() during gameplay
    # shape: (720, 1280, 3) uint8 — full game resolution

    H, W = input_size              # H=90, W=160

    img = Image.fromarray(obs)
          .resize((W, H), BILINEAR) # PIL takes (W,H) → resize to 160×90
    
    t = torch.from_numpy(np.array(img))
        .permute(2, 0, 1)           # (90, 160, 3) → (3, 90, 160)
        .float() / 255.0            # normalize to [0.0, 1.0]
    
    return t.unsqueeze(0)           # add batch dim → (1, 3, 90, 160)
    #                                  CNN expects (batch, channels, H, W)
    #                                  during training batch=64
    #                                  during gameplay batch=1 (one frame at a time)
```

Then `decode_detections` takes the CNN output and converts back to pixel boxes:

```python
def decode_detections(preds, conf_threshold=0.5, iou_threshold=0.4,
                       orig_w=1280, orig_h=720):
"""this is where the CNN's raw output becomes actual bounding boxes:"""

    # preds arrives as (1, 220, 5) — the 1 is just the batch wrapper, drop it
    preds_np = preds[0]   # now (220, 5), one row per grid cell

    # column 0 is confidence — how sure the CNN is that cell contains a zombie
    # keep only cells where CNN is at least 50% sure
    mask = preds_np[:, 0] >= conf_threshold
    detected = preds_np[mask]   # goes from 220 rows down to maybe 2 or 3

    if len(detected) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # CNN outputs coordinates normalized to [0,1] for resolution independence
    # multiply back by screen dimensions to get actual pixel positions
    boxes = np.stack([
        detected[:, 1] * orig_w,   # x: how far from left edge in pixels
        detected[:, 2] * orig_h,   # y: how far from top edge in pixels
        detected[:, 3] * orig_w,   # w: zombie width in pixels
        detected[:, 4] * orig_h,   # h: zombie height in pixels
    ], axis=1)

    # a zombie near a grid cell border gets detected by multiple adjacent cells
    # NMS keeps only the most confident box when two boxes overlap more than 40%
    keep = _nms(boxes, detected[:, 0], iou_threshold)
    return boxes[keep]

```

---

## Side by side — dataset vs preprocessing

```
                    dataset.py              preprocessing.py
                    ──────────              ────────────────
When used:          training offline        gameplay realtime
Input shape:        (180, 320, 3)           (720, 1280, 3)
Output shape:       (3, 90, 160) tensor     (1, 3, 90, 160) tensor
Batch dimension:    added by DataLoader     added manually (.unsqueeze(0))
Has labels:         yes → (8, 5) tensor     no labels, just the image
After CNN:          loss computed           decode_detections() called
Coordinates out:    normalized [0,1]        pixel space (1280×720)
```

Ready for `cnn.py` now? That's the missing middle — what happens between those input tensors and the 220 predictions that come out.