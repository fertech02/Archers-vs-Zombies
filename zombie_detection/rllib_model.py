import os
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import CNN_INPUT_SIZE


class KAZVisionModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})

        self.cnn = ZombieCNN(input_shape=(3, *CNN_INPUT_SIZE))
        self._cnn_frozen = False
        self._frame_stack = int(custom_cfg.get("frame_stack", 1))

        checkpoint = custom_cfg.get("cnn_checkpoint")
        if checkpoint and os.path.isfile(checkpoint):
            self.cnn.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            for p in self.cnn.parameters():
                p.requires_grad = False
            self.cnn.eval()
            self._cnn_frozen = True
            n_frozen = sum(p.numel() for p in self.cnn.parameters())
            print(f"[KAZVisionModel] Loaded + froze CNN from {checkpoint} "
                  f"({n_frozen:,} parameters frozen, frame_stack={self._frame_stack})")
        elif checkpoint:
            print(f"[KAZVisionModel] WARNING: checkpoint not found at {checkpoint} "
                  f"— CNN will be trained from scratch")

        feat = self.cnn.feat_size * self._frame_stack
        self.policy_head = nn.Linear(feat, num_outputs)
        self.value_head  = nn.Linear(feat, 1)
        self._features   = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self._cnn_frozen:
            self.cnn.eval()
        return self

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"].float() / 255.0
        B, H, W, C = obs.shape
        N = self._frame_stack
        assert C == 3 * N, f"expected {3*N} channels with frame_stack={N}, got {C}"

        obs = obs.permute(0, 3, 1, 2).contiguous().reshape(B * N, 3, H, W)
        obs = nn.functional.interpolate(
            obs, size=CNN_INPUT_SIZE, mode="bilinear", align_corners=False
        )

        if self._cnn_frozen:
            with torch.no_grad():
                feats = self.cnn.extract_features(obs)   # (B*N, 512)
        else:
            feats = self.cnn.extract_features(obs)

        self._features = feats.reshape(B, N * self.cnn.feseat_size)
        return self.policy_head(self._features), state

    def value_function(self):
        return self.value_head(self._features).squeeze(1)
