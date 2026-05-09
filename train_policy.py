"""
train_policy.py
---------------
Trains a PPO policy for the KAZ environment using RLlib.

Architecture: SimpleKAZModel — Nature DQN CNN + MLP.
No connection to the zombie detection CNN (kept fully separate).
distortion_level=0 for fast training (no transforms).
"""

import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pettingzoo.utils import aec_to_parallel
from ray.tune import CLIReporter

from utils import create_environment
from zombie_detection.rllib_model import SimpleKAZModel
from reward_wrapper import ShapedRewardWrapper

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ppo_kaz")
HERE        = os.path.dirname(os.path.abspath(__file__))


def make_env():
    aec = create_environment(
        distortion_level=0,
        render_mode=None,
        max_cycles=2500,
    )
    aec = ShapedRewardWrapper(aec)
    return ParallelPettingZooEnv(aec_to_parallel(aec))


def main():
    use_gpu = int(os.environ.get("USE_GPU", "0"))

    ray.init(
        num_gpus=use_gpu,
        ignore_reinit_error=True,
        object_store_memory=2_000_000_000,
        runtime_env={
            "working_dir": HERE,
            "excludes": ["results", ".git", "__pycache__",
                         "*.pth", "zombie_dataset"],
        },
    )

    register_env("kaz", lambda _: make_env())
    ModelCatalog.register_custom_model("kaz_simple", SimpleKAZModel)

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
            num_env_runners=4,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
            sample_timeout_s=300,
        )
        .training(
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=4,
            entropy_coeff=0.005,
            model={
                "custom_model": "kaz_simple",
                "custom_model_config": {
                    "cnn_checkpoint": os.path.join(HERE, "zombie_detection", "zombie_cnn.pth"),
                },
            },
        )
        .resources(
            num_gpus=use_gpu,
            num_cpus_for_main_process=1,
        )
        .framework("torch")
    )

    tune.run(
        "PPO",
        name="kaz_ppo_v3",
        config=config.to_dict(),
        stop={"training_iteration": 300},
        checkpoint_freq=5,
        checkpoint_at_end=True,
        storage_path=RESULTS_DIR,
        progress_reporter=reporter,
        verbose=2,
    )

    ray.shutdown()


reporter = CLIReporter(
    metric_columns={
        "training_iteration":  "iter",
        "episode_reward_mean": "rew_mean",
        "episode_reward_min":  "rew_min",
        "episode_reward_max":  "rew_max",
        "episode_len_mean":    "ep_len",
        "info/learner/default_policy/learner_stats/policy_loss": "pi_loss",
        "info/learner/default_policy/learner_stats/entropy":     "entropy",
    },
    max_report_frequency=10,
)

if __name__ == "__main__":
    main()
