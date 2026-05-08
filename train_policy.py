"""
train_policy.py
---------------
Trains a PPO policy for the KAZ environment using RLlib.

Changes vs teammate's baseline:
  1. Distortion randomized 0-5 per worker episode
     → policy learns to play under all visual conditions, not just level 0
  2. ShapedRewardWrapper applied during training only
     → danger urgency, survival, zone coordination, zombie count pressure
  3. 300 iterations instead of 500
     → more experience for complex behaviors to emerge
  4. Larger train_batch_size (2048) for A100
     → more stable gradient updates per iteration
  5. num_env_runners set conservatively for Colab CPU count
     → GPU handles network updates, CPUs run environments
"""

import os
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pettingzoo.utils import aec_to_parallel
from ray.tune import CLIReporter

from utils import create_environment
from zombie_detection.rllib_model import KAZVisionModel
from reward_wrapper import ShapedRewardWrapper

HERE           = os.path.dirname(os.path.abspath(__file__))
CNN_CHECKPOINT = os.path.join(HERE, "zombie_detection", "zombie_cnn.pth")
RESULTS_DIR    = os.path.join(HERE, "results", "ppo_kaz")
FRAME_STACK    = 4


def make_env():
    # randomize distortion level per environment instance
    # each of the parallel workers sees a different level → robust policy
    level = int(np.random.randint(0, 6))

    aec = create_environment(
        distortion_level=level,
        frame_stack=FRAME_STACK,
        render_mode=None,
        max_cycles=500,
    )

    # apply reward shaping — training only, never in submission
    aec = ShapedRewardWrapper(aec)

    return ParallelPettingZooEnv(aec_to_parallel(aec))


def main():
    # 1 su Colab/A100, 0 su macchine senza GPU (override via env var USE_GPU=1)
    use_gpu = int(os.environ.get("USE_GPU", "0"))

    ray.init(
        num_gpus=use_gpu,
        ignore_reinit_error=True,
        object_store_memory=20_000_000_00,  # limit object store to 2GB
        runtime_env={
            "working_dir": HERE,
            "excludes": ["results", ".git", "__pycache__",
                         "*.pth", "zombie_dataset"],
        },
    )

    register_env("kaz", lambda _: make_env())
    ModelCatalog.register_custom_model("kaz_vision", KAZVisionModel)

    # probe env for obs/action spaces
    tmp_env = make_env()
    tmp_env.close()

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("kaz")
        .env_runners(
            num_env_runners=4,           # CPU-bound — 4 is safe for Colab
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
            sample_timeout_s=300,    # give workers 5 minutes instead of 60s
        )
        .training(
            train_batch_size=2048,       # larger batch for A100 stability
            minibatch_size=128,
            num_epochs=10,
            # slightly higher entropy encourages exploration early on
            # helps agent discover positioning + aiming strategies
            entropy_coeff=0.02,
            model={
                "custom_model": "kaz_vision",
                "custom_model_config": {
                    "cnn_checkpoint": CNN_CHECKPOINT,
                    "frame_stack":    FRAME_STACK,
                },
            },
        )
        .resources(
            num_gpus=use_gpu,            # A100 if available, else CPU-only
            num_cpus_for_main_process=1,
        )
        .framework("torch")
    )

    tune.run(
        "PPO",
        name="kaz_ppo_v2",
        config=config.to_dict(),
        stop={"training_iteration": 300},
        checkpoint_freq=5,
        checkpoint_at_end=True,
        storage_path=RESULTS_DIR,
        progress_reporter=reporter,
        verbose=2,
    )

    """tune.run(
        "PPO",
        name="kaz_ppo",
        config=config.to_dict(),
        stop={"training_iteration": 300},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        storage_path=RESULTS_DIR,
        resume=True,
        verbose=1,
    )
    """
    ray.shutdown()

reporter = CLIReporter(
    metric_columns={
        "training_iteration": "training_iteration",
        "episode_reward_mean": "rew_mean",
        "episode_reward_min": "rew_min",
        "episode_reward_max": "rew_max",
        "episode_len_mean": "episode_len_mean",
        "info/learner/default_policy/learner_stats/policy_loss" : "pi_loss",
        "info/learner/default_policy/learner_stats/entropy" : "entropy",
        "info/learner/default_policy/learner_stats/kl" : "kl",
    },
    max_report_frequency=10,
)

if __name__ == "__main__":
    main()