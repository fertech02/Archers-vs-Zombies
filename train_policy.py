"""
train_policy.py
---------------
Trains a PPO MLP policy on the 32-dim KAZ feature vector.

Pipeline:
  KAZ (pixels) -> VectorObsWrapper (privileged 32-dim vector)
                -> PPO + VectorMLPPolicy

At submission time, the same 32-dim vector is built from CNN-detected zombies
and env.agent_list (see submission.py).
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils import aec_to_parallel
from ray.tune import CLIReporter

from utils import create_environment
from vector_obs_wrapper import VectorObsWrapper
from vector_policy import VectorMLPPolicy

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ppo_kaz")
HERE        = os.path.dirname(os.path.abspath(__file__))


def make_env():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = knights_archers_zombies_v10.env(
        max_cycles=2500,
        num_archers=2,
        num_knights=0,
        max_zombies=4,
        vector_state=False,
        render_mode=None,
    )
    env = ss.black_death_v3(env)
    env = VectorObsWrapper(env)
    return ParallelPettingZooEnv(aec_to_parallel(env))


def main():
    use_gpu = 0

    ray.init(
        num_gpus=use_gpu,
        ignore_reinit_error=True,
        object_store_memory=1_000_000_000,
        runtime_env={
            "working_dir": HERE,
            "excludes": ["results", ".git", "__pycache__",
                         "*.pth", "zombie_dataset"],
        },
    )

    register_env("kaz", lambda _: make_env())
    ModelCatalog.register_custom_model("vector_mlp", VectorMLPPolicy)

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
            sample_timeout_s=600,
        )
        .training(
            train_batch_size=10000,
            lr=3e-4,
            minibatch_size=1000,
            num_epochs=4,
            entropy_coeff=0.03,
            grad_clip=0.5,
            model={
                "custom_model": "vector_mlp",
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
        name="kaz_ppo_vector",
        config=config.to_dict(),
        stop={"training_iteration": 1000}, 
        checkpoint_freq=10,
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