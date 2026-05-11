"""
submission_privileged.py — diagnostic only.

Same MLP policy as submission.py but reads zombie positions directly from
env.zombie_list (privileged game state) instead of the CNN.
This bypasses the CNN entirely to isolate whether the reward gap is caused
by CNN detection errors or by something else (reward scale, episode length, etc.).
"""
import os
from pathlib import Path
from typing import Callable

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

from vector_policy import VectorMLPPolicy
from vector_obs_wrapper import build_vector, VECTOR_DIM

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_POLICY_PATH = _HERE / "policy.pth"

ENV_SETTINGS = {
    "distortion_level": 0,
}


class CustomWrapper(BaseWrapper):
    """Identity wrapper."""
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)


class CustomPredictFunction(Callable):
    """
    Privileged diagnostic: reads zombie positions from env.zombie_list directly,
    bypassing the CNN. Everything else is identical to submission.py.
    """

    def __init__(self, env: gymnasium.Env):
        self.env = env

        vector_space = spaces.Box(
            low=-2.0, high=2.0, shape=(VECTOR_DIM,), dtype=np.float32
        )
        agent_id = list(env.possible_agents)[0] if hasattr(env, "possible_agents") else "archer_0"
        action_space = env.action_space(agent_id)

        self.model = VectorMLPPolicy(
            obs_space=vector_space,
            action_space=action_space,
            num_outputs=action_space.n,
            model_config={},
            name="vector_mlp",
        )

        if _POLICY_PATH.exists():
            self.model.load_state_dict(
                torch.load(str(_POLICY_PATH), map_location="cpu"), strict=True
            )
            print(f"[privileged] Loaded policy from {_POLICY_PATH}")
        else:
            print(f"[privileged] WARNING: policy.pth not found, using random weights")
        self.model.eval()

    def __call__(self, observation, agent, *args, **kwargs):
        try:
            game = self.env.unwrapped
            idx = game.agent_name_mapping[agent]
            my_archer = game.agent_list[idx]

            teammate_archer = None
            for j, other in enumerate(game.agent_list):
                if j != idx:
                    teammate_archer = other
                    break

            # Privileged: exact positions from game state, no CNN
            zombie_positions = [
                (z.rect.centerx, z.rect.centery, z.rect.width, z.rect.height)
                for z in game.zombie_list
            ]
        except Exception:
            class _Dummy:
                class rect:
                    centerx = 640
                    centery = 360
                direction = (0.0, -1.0)
            my_archer = _Dummy()
            teammate_archer = None
            zombie_positions = []

        vec = build_vector(my_archer, teammate_archer, zombie_positions)
        obs_t = torch.FloatTensor(vec).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model({"obs": obs_t}, [], None)
        return int(torch.argmax(logits, dim=-1).item())


class CustomZombieDetectorFunction(Callable):
    """Stub — not used in this diagnostic."""
    def __init__(self, env):
        pass

    def __call__(self, observation, *args, **kwargs):
        return np.zeros((0, 4), dtype=np.float32)
