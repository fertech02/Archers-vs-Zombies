import os
from typing import Callable

import gymnasium
import torch
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import decode_detections, preprocess_obs

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "zombie_detection", "zombie_cnn.pth")

_ZOMBIE_W_NORM = 7.25 / 320
_ZOMBIE_H_NORM = 7.75 / 180


class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)


class CustomPredictFunction(Callable):
    """Function to use to load the trained model and predict the action"""
    def __init__(self, env):
        ckpt = (Path(os.path.dirname(os.path.abspath(__file__)))
                / "results" / "ppo_kaz" / "<run_dir>" / "checkpoint_xxx"
                / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(ckpt)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]  # "shared_policy" è condivisa
        fwd_in = {"obs": torch.Tensor(observation).unsqueeze(0)}
        out = rl_module.forward_inference(fwd_in)
        dist_cls = rl_module.get_inference_action_dist_cls()
        return dist_cls.from_logits(out["action_dist_inputs"]).sample()[0].numpy()


class CustomZombieDetectorFunction(Callable):
    """Function to use to load the trained model and predict where
    the zombies are.
    """

    def __init__(self, env: gymnasium.Env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ZombieCNN(input_shape=(3, 90, 160))
        self.model.load_state_dict(torch.load(_MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, observation, *args, **kwargs):

        if observation.ndim == 1:
            n_pixels = observation.size // 3
            orig_h = int((n_pixels * 9 / 16) ** 0.5)
            orig_w = n_pixels // orig_h
            observation = observation.reshape(orig_h, orig_w, 3)
        orig_h, orig_w = observation.shape[:2]
        tensor = preprocess_obs(observation).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)

        boxes = decode_detections(preds, conf_threshold=0.5, orig_w=orig_w, orig_h=orig_h)
        if len(boxes) > 0:
            boxes[:, 2] = _ZOMBIE_W_NORM * orig_w
            boxes[:, 3] = _ZOMBIE_H_NORM * orig_h
        return boxes
