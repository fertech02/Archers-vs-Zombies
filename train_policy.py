import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from pettingzoo.utils import aec_to_parallel

from utils import create_environment
from zombie_detection.rllib_model import KAZVisionModel

HERE = os.path.dirname(os.path.abspath(__file__))
CNN_CHECKPOINT = os.path.join(HERE, "zombie_detection", "zombie_cnn.pth")
RESULTS_DIR = os.path.join(HERE, "results", "ppo_kaz")

DISTORTION = 0
FRAME_STACK = 4


def make_env():
    level = DISTORTION
    aec = create_environment(
        distortion_level=level,
        frame_stack=FRAME_STACK,
        render_mode=None,
    )
    return ParallelPettingZooEnv(aec_to_parallel(aec))


def main():
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": os.path.dirname(os.path.abspath(__file__)),
            "excludes": ["results", ".git", "__pycache__", "*.pth"]  # Escludi i pesi qui
        }
    )
    register_env("kaz", lambda _: make_env())
    ModelCatalog.register_custom_model("kaz_vision", KAZVisionModel)

    tmp_env = make_env()
    obs_space = tmp_env.observation_space
    action_space = tmp_env.action_space
    tmp_env.close()

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("kaz")
        .env_runners(
            num_env_runners=8,  # Scendiamo a 8 per salvare la RAM
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=2048,  # Dimensione bilanciata per 8 worker
            minibatch_size=128,
            num_epochs=10,
            model={
                "custom_model": "kaz_vision",
                "custom_model_config": {
                    "cnn_checkpoint": CNN_CHECKPOINT,

                    "frame_stack": 4,
                },
            }
        )
        .resources(
            num_cpus_for_main_process=2,  # Lascia spazio al driver
        )
        .framework("torch")
    )

    tune.run(
        "PPO",
        name="kaz_ppo",
        config=config.to_dict(),
        stop={"training_iteration": 500},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        storage_path=RESULTS_DIR,
        verbose=1,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()