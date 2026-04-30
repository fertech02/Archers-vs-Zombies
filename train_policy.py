"""
Train a PPO policy on KnightsArchersZombies (PettingZoo) using the
pretrained ZombieCNN backbone as a visual feature extractor.

Usage:
    python train_policy.py

Requires:
    pip install ray[rllib] pettingzoo[butterfly]
"""
import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from pettingzoo.butterfly import knights_archers_zombies_v10

from zombie_detection.rllib_model import KAZVisionModel

CNN_CHECKPOINT = os.path.join("zombie_detection", "zombie_cnn.pth")
RESULTS_DIR    = os.path.abspath(os.path.join("results", "ppo_kaz"))


def make_env():
    return ParallelPettingZooEnv(knights_archers_zombies_v10.parallel_env(render_mode=None))


def main():
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": os.path.dirname(os.path.abspath(__file__))},
        # Surface worker crash logs instead of swallowing them
        log_to_driver=True,
    )

    register_env("kaz", lambda _: make_env())
    ModelCatalog.register_custom_model("kaz_vision", KAZVisionModel)

    # sample one env to get obs/action spaces
    tmp_env = make_env()
    obs_space    = tmp_env.observation_space
    action_space = tmp_env.action_space
    tmp_env.close()

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("kaz")
        .framework("torch")
        # Start with 0 remote workers (runs env in driver process).
        # Increase to 2+ once confirmed working.
        .env_runners(num_env_runners=0, rollout_fragment_length=128)
        .training(
            model={
                "custom_model": "kaz_vision",
                "custom_model_config": {
                    "cnn_checkpoint": CNN_CHECKPOINT,
                },
            },
            train_batch_size=2048,
            minibatch_size=256,
            num_epochs=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, action_space, {})
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .resources(num_gpus=int(os.environ.get("NUM_GPUS", 0)))
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
