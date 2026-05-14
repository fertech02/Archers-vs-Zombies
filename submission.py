"""
At submission time:
  1. observation arrives as pixels (per professor's mandate)
  2. CNN detects zombies from pixels -> bounding boxes
  3. archer state for both archers read from env.agent_list (explicitly allowed)
  4. boxes + archer state -> 48-dim vector -> MLP policy -> action

The MLP was trained on a 48-dim vector built from privileged game state.
"""
import os
import sys
from pathlib import Path
from typing import Callable
import gymnasium
import numpy as np
import torch

from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import decode_detections, preprocess_obs
from vector_policy import VectorMLPPolicy
from vector_obs_wrapper import build_vector, VECTOR_DIM

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MODEL_PATH  = HERE / "zombie_detection" / "zombie_cnn.pth"
POLICY_PATH = HERE / "policy.pth"

ZOMBIE_W_NORM = 7.25 / 320
ZOMBIE_H_NORM = 7.75 / 180

class CustomWrapper(BaseWrapper):
    """Identity wrapper — CustomPredictFunction does the real work."""
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)


class CustomPredictFunction(Callable):

    """
    Pixels (observation arg) -> CNN -> 48-dim vector -> MLP -> action.
    """
    def __init__(self, env: gymnasium.Env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN for zombie detection (pixels -> boxes)
        self.cnn = ZombieCNN(input_shape=(3, 90, 160))
        # Load pre-trained weights
        self.cnn.load_state_dict(torch.load(str(MODEL_PATH), map_location=self.device))
        self.cnn.to(self.device)
        # Inference
        self.cnn.eval()

        # MLP policy
        vector_space = spaces.Box(
            low=-2.0, high=2.0, shape=(VECTOR_DIM,), dtype=np.float32
        )
        agent_id = list(env.possible_agents)[0] if hasattr(env, "possible_agents") else "archer_0"
        action_space = env.action_space(agent_id)

        # Init MLP
        self.model = VectorMLPPolicy(
            obs_space=vector_space,
            action_space=action_space,
            num_outputs=action_space.n,
            model_config={},
            name="vector_mlp",
        )

        # Loads Policy weights
        if POLICY_PATH.exists():
            self.model.load_state_dict(
                torch.load(str(POLICY_PATH), map_location="cpu"), strict=True
            )
        # Inference
        self.model.eval()


    def __call__(self, observation, agent, *args, **kwargs):

        """
            Observations arrives as pixels -> return an int
        """
        # If observation is flat -> reconstruct dimensions
        if observation.ndim == 1:
            n_pixels = observation.size // 3
            orig_h = int((n_pixels * 9 / 16) ** 0.5)
            orig_w = n_pixels // orig_h
            observation = observation.reshape(orig_h, orig_w, 3)
        orig_h, orig_w = observation.shape[:2]

        # CNN -> zombie boxes -> (x_center, y_center) tuples
        tensor = preprocess_obs(observation).to(self.device)
        with torch.no_grad():
            preds = self.cnn(tensor)
        boxes = decode_detections(preds, conf_threshold=0.7, orig_w=orig_w, orig_h=orig_h)
        w = ZOMBIE_W_NORM * orig_w
        h = ZOMBIE_H_NORM * orig_h
        # Gives the zombie position in the original space
        zombie_positions = [(b[0] + w / 2, b[1] + h / 2, w, h) for b in boxes]

        # Both archers from env.agent_list (allowed)
        try:
            game = self.env.unwrapped
            idx = game.agent_name_mapping[agent]
            my_archer = game.agent_list[idx]

            teammate_archer = None
            for j, other in enumerate(game.agent_list):
                if j != idx:
                    teammate_archer = other
                    break
        except Exception:
            """
                If we are not able to access the state, we build a vector
                with direction (0.0, -1.0).
            """
            class _Dummy:
                class rect:
                    centerx = 640
                    centery = 360
                direction = (0.0, -1.0)
            my_archer = _Dummy()
            teammate_archer = None

        # Same build_vector function used during training
        vec = build_vector(my_archer, teammate_archer, zombie_positions)

        # Build torch tensor
        obs_t = torch.FloatTensor(vec).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model({"obs": obs_t}, [], None)

        # Builds action probability distribution. Applies softmax internally.
        dist = torch.distributions.Categorical(logits=logits)
        # Stochastic sampling, keeps exploration
        return int(dist.sample().item())

class CustomZombieDetectorFunction(Callable):

    def __init__(self, env: gymnasium.Env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ZombieCNN(input_shape=(3, 90, 160))
        self.model.load_state_dict(torch.load(str(MODEL_PATH), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, observation, *args, **kwargs):

        # If observation is flat -> reconstruct dimensions
        if observation.ndim == 1:
            n_pixels = observation.size // 3
            orig_h = int((n_pixels * 9 / 16) ** 0.5)
            orig_w = n_pixels // orig_h
            observation = observation.reshape(orig_h, orig_w, 3)
        orig_h, orig_w = observation.shape[:2]

        tensor = preprocess_obs(observation).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)

        boxes = decode_detections(preds, conf_threshold=0.6, orig_w=orig_w, orig_h=orig_h)
        if len(boxes) > 0:
            boxes[:, 2] = ZOMBIE_W_NORM * orig_w
            boxes[:, 3] = ZOMBIE_H_NORM * orig_h
        return boxes