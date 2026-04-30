"""
RLlib TorchModelV2 that wraps ZombieCNN as a visual feature extractor.

The CNN backbone extracts a 512-dim feature vector from each pixel observation.
That vector feeds into separate policy and value heads (for PPO / actor-critic).

Registration example (in your training script):
    from ray.rllib.models import ModelCatalog
    from zombie_detection.rllib_model import KAZVisionModel
    ModelCatalog.register_custom_model("kaz_vision", KAZVisionModel)

Then in the algorithm config:
    .training(model={
        "custom_model": "kaz_vision",
        "custom_model_config": {
            "cnn_checkpoint": "zombie_detection/zombie_cnn.pth",  # optional
        },
    })
"""
import os
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import CNN_INPUT_SIZE


class KAZVisionModel(TorchModelV2, nn.Module):
    """
    Pixel observation → CNN backbone → 512-dim features → policy / value heads.

    The CNN can optionally be initialised from a checkpoint trained on the
    zombie detection task (see zombie_detection/train.py).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})

        self.cnn = ZombieCNN(input_shape=(3, *CNN_INPUT_SIZE))

        checkpoint = custom_cfg.get("cnn_checkpoint")
        if checkpoint and os.path.isfile(checkpoint):
            self.cnn.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            print(f"[KAZVisionModel] Loaded pretrained CNN from {checkpoint}")

        feat = self.cnn.feat_size          # 512
        self.policy_head = nn.Linear(feat, num_outputs)
        self.value_head  = nn.Linear(feat, 1)
        self._features   = None            # cached for value_function()

    def forward(self, input_dict, state, seq_lens):
        """
        input_dict["obs"]: (B, H, W, C) uint8 — raw environment observation
        Returns action logits of shape (B, num_outputs).
        """
        obs = input_dict["obs"].float() / 255.0          # normalize to [0,1]
        obs = obs.permute(0, 3, 1, 2)                    # (B,H,W,C) → (B,C,H,W)
        obs = nn.functional.interpolate(
            obs, size=CNN_INPUT_SIZE, mode="bilinear", align_corners=False
        )
        self._features = self.cnn.extract_features(obs)  # (B, 512)
        return self.policy_head(self._features), state

    def value_function(self):
        return self.value_head(self._features).squeeze(1)
