import argparse
import ray
from ray import tune
from ray.tune.registry import register_env

from ray.tune.logger import pretty_print, DEFAULT_LOGGERS, TBXLogger

# from ray.tune.integration.wandb import WandbLogger

from environment import DemoMultiAgentEnv
from model_for_global import Model
from ray.rllib.models import ModelCatalog
from multi_trainer import MultiPPOTrainer
from multi_action_dist import TorchHomogeneousMultiActionDistribution
from flatland_environment_global import FlatlandMultiAgentEnv
def on_episode_end(info):
    episode = info["episode"]

    assert len(episode._agent_to_last_info) == 1
    agent_info = next(iter(episode._agent_to_last_info.values()))

    episode_max_steps = agent_info["max_episode_steps"]
    episode_num_agents = agent_info["num_agents"]
    episode_steps = 0
    episode_score = 0
    episode_done_agents = 0

    for i in agent_info["rewards"]:
        episode_steps = max(episode_steps, agent_info["agent_step"][i])
        episode_score += agent_info["agent_score"][i]
        episode_done_agents += agent_info["done"][i]

    norm_factor = 1.0 / (episode_max_steps * episode_num_agents)
    percentage_complete = float(episode_done_agents) / episode_num_agents

    episode.custom_metrics["episode_steps"] = episode_steps
    episode.custom_metrics["episode_max_steps"] = episode_max_steps
    episode.custom_metrics["episode_num_agents"] = episode_num_agents
    episode.custom_metrics["episode_return"] = episode.total_reward
    episode.custom_metrics["episode_score"] = episode_score
    episode.custom_metrics["episode_score_normalized"] = episode_score * norm_factor
    episode.custom_metrics["percentage_complete"] = percentage_complete
    episode.custom_metrics["agent_deadlocks"] = agent_info["agent_deadlock"]

def train(share_observations=True, action_space="discrete", goal_shift=1):
    ray.init()

    register_env("demo_env", lambda config: FlatlandMultiAgentEnv(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )
 
    tune.run(
        MultiPPOTrainer,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        local_dir="/tmp",
        # loggers=DEFAULT_LOGGERS + (WandbLogger,),
        stop={"timesteps_total": 10000000},
        config={
            "framework": "torch",
            "env": "demo_env",
            "kl_coeff": 0.0,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "train_batch_size": 1000,
            "sgd_minibatch_size": 100,
            "num_sgd_iter": 10,
            "num_gpus": 0,
            "num_workers": 6,
            "num_envs_per_worker": 5,
            "lr": 5e-4,
            "gamma": 0.99,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "model": {
                "custom_model": "model",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "encoder_out_features": 50,
                    "shared_nn_out_features_per_agent": 50,
                    "value_state_encoder_cnn_out_features": 16,
                    "share_observations": share_observations,
                },
            },
            "callbacks": {
                "on_episode_end": on_episode_end,
            },
            "logger_config": {
                "wandb": {
                    "project": "ray_multi_agent_trajectory",
                    "group": "a",
                    "api_key_file": "./wandb_api_key_file",
                }
            },


            "env_config": {
                'x_dim': 25,
                'y_dim': 25,
                'n_cities': 3,
                'max_rails_between_cities': 2,
                'sparse_reward': True,
                'max_rails_in_city': 3,
                'observation_max_path_depth': 30,
                'observation_tree_depth': 2,
                'observation_radius': 10,
                'n_agents': 5,
                "world_shape": [1, 1],
                'render': False
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLLib multi-agent with shared NN demo."
    )
    parser.add_argument(
        "--action_space",
        default="discrete",
        const="discrete",
        nargs="?",
        choices=["continuous", "discrete"],
        help="Train with continuous or discrete action space",
    )
    parser.add_argument(
        "--disable_sharing",
        action="store_true",
        help="Do not instantiate shared central NN for sharing information",
    )
    parser.add_argument(
        "--goal_shift",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Goal shift offset (0 means that each agent moves to its own goal, 1 to its neighbor, etc.)",
    )

    args = parser.parse_args()
    train(
        share_observations=not args.disable_sharing,
        action_space=args.action_space,
        goal_shift=args.goal_shift,
    )
