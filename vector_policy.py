"""
vector_policy.py
----------------
Small MLP policy operating on the 32-dim feature vector.
Architecture: shared trunk -> policy head + value head (PPO actor-critic).
"""
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class VectorMLPPolicy(TorchModelV2, nn.Module):
    """
    32-dim input -> 64 -> 64 -> (policy logits, value).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        feat_dim = obs_space.shape[0]  # 32

        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, num_outputs)
        self.value_head = nn.Linear(128, 1)
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        self._features = self.shared(obs)
        return self.policy_head(self._features), state

    def value_function(self):
        return self.value_head(self._features).squeeze(1)