from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from zombie_detection.cnn import ZombieCNN

_DEFAULT_CNN_PATH = Path(__file__).parent / "zombie_cnn.pth"


class SimpleKAZModel(TorchModelV2, nn.Module):
    """
    RL policy using frozen ZombieCNN conv layers as visual backbone.
    Backbone is fixed; only policy_head and value_head are trained by RL.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cnn_path = model_config.get("custom_model_config", {}).get(
            "cnn_checkpoint", str(_DEFAULT_CNN_PATH)
        )

        zombie_cnn = ZombieCNN(input_shape=(3, 90, 160))
        if Path(cnn_path).exists():
            zombie_cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))

        self.backbone = zombie_cnn.backbone  # frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Trainable adapter: adapts detection features to policy-useful features (SpawnNet, Lin et al. 2023)
        self.adapter = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((3, 5))

        feat_size = 32 * 3 * 5  # 480

        self.policy_head = nn.Sequential(
            nn.Linear(feat_size, 256), nn.ReLU(),
            nn.Linear(256, num_outputs),
        )
        self.value_head = nn.Sequential(
            nn.Linear(feat_size, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float() / 255.0
        obs = obs.permute(0, 3, 1, 2)
        obs = F.interpolate(obs, size=(90, 160), mode="bilinear", align_corners=False)
        with torch.no_grad():
            backbone_out = self.backbone(obs)
        self._features = self.pool(self.adapter(backbone_out)).flatten(1)  # (B, 480)
        return self.policy_head(self._features), state

    def value_function(self):
        return self.value_head(self._features).squeeze(1)
