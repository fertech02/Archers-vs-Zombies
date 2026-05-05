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

HERE           = os.path.dirname(os.path.abspath(__file__))
CNN_CHECKPOINT = os.path.join(HERE, "zombie_detection", "zombie_cnn.pth")
RESULTS_DIR    = os.path.join(HERE, "results", "ppo_kaz")

DISTORTION  = 0
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
        runtime_env={"working_dir": os.path.dirname(os.path.abspath(__file__))},
        log_to_driver=True,
    )

    register_env("kaz", lambda _: make_env())
    ModelCatalog.register_custom_model("kaz_vision", KAZVisionModel)

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
        .env_runners(num_env_runners=2, rollout_fragment_length=128)
        .training(
            model={
                "custom_model": "kaz_vision",
                "custom_model_config": {
                    "cnn_checkpoint": CNN_CHECKPOINT,
                    "frame_stack":    FRAME_STACK,
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
            grad_clip=0.5
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
