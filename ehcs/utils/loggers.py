import gymnasium as gym
import numpy as np
import wandb
from sinergym.utils.common import is_wrapped
from sinergym.utils.wrappers import NormalizeObservation


class EvalLogger:
    def __init__(self, obs_variables, action_variables, writer):
        self.obs_variables = obs_variables
        self.action_variables = action_variables
        self.writer = writer

        self.observation_data = {obs_name: [] for obs_name in self.obs_variables}
        self.action_data = {action_name: [] for action_name in self.action_variables}
        self.episodes = 0
        self.episode_data = {
            "rewards": [],
            "powers": [],
            "comfort_penalties": [],
            "power_penalties": [],
            "abs_comfort": [],
            "num_comfort_violation": 0,
            "timesteps": 0,
        }

    def log_update(self, observation, termination, info, global_step):
        # Log observations.
        for var_, val in zip(self.obs_variables, observation):
            self.observation_data[var_].append(val)

        
        # Log actions.
        for var_, val in zip(self.action_variables, info["action"]):
            self.action_data[var_].append(val) # Actual action.

        # Log info.
        self.episode_data["rewards"].append(info["reward"])
        self.episode_data["powers"].append(info["total_energy"])
        self.episode_data["comfort_penalties"].append(info["reward_comfort"])
        self.episode_data["power_penalties"].append(info["reward_energy"])
        self.episode_data["abs_comfort"].append(info["abs_comfort"])
        if info["reward_comfort"] != 0:
            self.episode_data["num_comfort_violation"] += 1
        self.episode_data["timesteps"] += 1
        self.episode_data["time_elapsed"] = info["time_elapsed"]

    def log_episode(self, global_step):
        self.episodes += 1
        episode_metrics = {}
        # Reward.
        episode_metrics["mean_reward"] = np.mean(self.episode_data["rewards"])
        # Timesteps.
        episode_metrics["episode_length"] = self.episode_data["timesteps"]
        # Power.
        episode_metrics["mean_power"] = np.mean(self.episode_data["powers"])
        episode_metrics["cumulative_power"] = np.sum(self.episode_data["powers"])
        # Comfort Penalty.
        episode_metrics["mean_comfort_penalty"] = np.mean(
            self.episode_data["comfort_penalties"]
        )
        episode_metrics["cumulative_comfort_penalty"] = np.sum(
            self.episode_data["comfort_penalties"]
        )
        # Power Penalty.
        episode_metrics["mean_power_penalty"] = np.mean(
            self.episode_data["power_penalties"]
        )
        episode_metrics["cumulative_power_penalty"] = np.sum(
            self.episode_data["power_penalties"]
        )
        episode_metrics["episode_num"] = self.episodes

        try:
            episode_metrics["comfort_violation_time(%)"] = (
                self.episode_data["num_comfort_violation"]
                / self.episode_data["timesteps"]
                * 100
            )
        except ZeroDivisionError:
            episode_metrics["comfort_violation_time(%)"] = np.nan

        for key, metric in episode_metrics.items():
            self.writer.add_scalar(f"episode/{key}", metric, global_step)

        # Reset data for episode.
        self.episode_data = {
            "rewards": [],
            "powers": [],
            "comfort_penalties": [],
            "power_penalties": [],
            "abs_comfort": [],
            "num_comfort_violation": 0,
            "timesteps": 0,
        }

        # Observation data.
        episode_observations = {}
        for key, val in self.observation_data.items():
            episode_observations[key] = np.mean(val)

        for key, metric in episode_observations.items():
                self.writer.add_scalar(f"observations/{key}", metric, global_step)

        # Action data.
        episode_actions = {}
        for key, val in self.action_data.items():
            episode_actions[key] = np.mean(val)

        for key, metric in episode_actions.items():
                self.writer.add_scalar(f"actions/{key}", metric, global_step)

        # Reset values.
        self.observation_data = {obs_name: [] for obs_name in self.obs_variables}
        self.action_data = {action_name: [] for action_name in self.action_variables}


class TrainLogger:
    def __init__(self, env_names, obs_variables, action_variables, writer):
        self.obs_variables = obs_variables
        self.action_variables = action_variables
        self.env_names = env_names
        self.writer = writer

        self.observation_data =  [{obs_name: [] for obs_name in self.obs_variables} for _ in range(len(env_names))]
        self.action_data =  [{action_name: [] for action_name in self.action_variables} for _ in range(len(env_names))]
        self.episodes = [0 for _ in range(len(env_names))]
        self.episode_data = [
            {
                "rewards": [],
                "powers": [],
                "comfort_penalties": [],
                "power_penalties": [],
                "abs_comfort": [],
                "num_comfort_violation": 0,
                "timesteps": 0,
            }
                for _ in range(len(env_names))
        ]

    def log_update(self, unwrapped_observations, terminations, infos, global_step):
        # Split infos into separate dictionaries.
        # Source: https://stackoverflow.com/a/1780295
        split_infos = list(
            map(dict, zip(*[[(k, v) for v in value] for k, value in infos.items()]))
        )

        for i, env_name in enumerate(self.env_names):
            unwrapped_obs = unwrapped_observations[i]
            terminated = terminations[i]
            info = split_infos[i]

            if "final_info" in info:
                info = info["final_info"]

            # Log observations.
            for var_, val in zip(self.obs_variables, unwrapped_obs):
                self.observation_data[i][var_].append(val)

            
            # Log actions.
            for var_, val in zip(self.action_variables, info["action"]):
                self.action_data[i][var_].append(val) # Actual action.

            # Log info.
            self.episode_data[i]["rewards"].append(info["reward"])
            self.episode_data[i]["powers"].append(info["total_energy"])
            self.episode_data[i]["comfort_penalties"].append(info["reward_comfort"])
            self.episode_data[i]["power_penalties"].append(info["reward_energy"])
            self.episode_data[i]["abs_comfort"].append(info["abs_comfort"])
            if info["reward_comfort"] != 0:
                self.episode_data[i]["num_comfort_violation"] += 1
            self.episode_data[i]["timesteps"] += 1
            self.episode_data[i]["time_elapsed"] = info["time_elapsed"]

    def log_episode(self, idx, global_step):
        env_name = self.env_names[idx]
        self.episodes[idx] += 1
        episode_metrics = {}
        # Reward.
        episode_metrics["mean_reward"] = np.mean(self.episode_data[idx]["rewards"])
        # Timesteps.
        episode_metrics["episode_length"] = self.episode_data[idx]["timesteps"]
        # Power.
        episode_metrics["mean_power"] = np.mean(self.episode_data[idx]["powers"])
        episode_metrics["cumulative_power"] = np.sum(self.episode_data[idx]["powers"])
        # Comfort Penalty.
        episode_metrics["mean_comfort_penalty"] = np.mean(
            self.episode_data[idx]["comfort_penalties"]
        )
        episode_metrics["cumulative_comfort_penalty"] = np.sum(
            self.episode_data[idx]["comfort_penalties"]
        )
        # Power Penalty.
        episode_metrics["mean_power_penalty"] = np.mean(
            self.episode_data[idx]["power_penalties"]
        )
        episode_metrics["cumulative_power_penalty"] = np.sum(
            self.episode_data[idx]["power_penalties"]
        )
        episode_metrics["episode_num"] = self.episodes[idx]

        try:
            episode_metrics["comfort_violation_time(%)"] = (
                self.episode_data[idx]["num_comfort_violation"]
                / self.episode_data[idx]["timesteps"]
                * 100
            )
        except ZeroDivisionError:
            episode_metrics["comfort_violation_time(%)"] = np.nan

        for key, metric in episode_metrics.items():
            self.writer.add_scalar(f"{env_name}/episode/{key}", metric, global_step)

        # Reset data for episode.
        self.episode_data[idx] = {
            "rewards": [],
            "powers": [],
            "comfort_penalties": [],
            "power_penalties": [],
            "abs_comfort": [],
            "num_comfort_violation": 0,
            "timesteps": 0,
        }

        # Observation data.
        episode_observations = {}
        for key, val in self.observation_data[idx].items():
            episode_observations[key] = np.mean(val)

        for key, metric in episode_observations.items():
                self.writer.add_scalar(f"observations/{key}", metric, global_step)

        # Action data.
        episode_actions = {}
        for key, val in self.action_data[idx].items():
            episode_actions[key] = np.mean(val)

        for key, metric in episode_actions.items():
                self.writer.add_scalar(f"actions/{key}", metric, global_step)

        # Reset values.
        self.observation_data[idx] = {obs_name: [] for obs_name in self.obs_variables}
        self.action_data[idx] = {action_name: [] for action_name in self.action_variables}
