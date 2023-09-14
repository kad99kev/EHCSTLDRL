import gymnasium as gym
from sinergym.utils.constants import (
    RANGES_5ZONE,
    RANGES_DATACENTER,
    RANGES_OFFICE,
    RANGES_OFFICEGRID,
    RANGES_SHOP,
    RANGES_WAREHOUSE,
)
from sinergym.utils.wrappers import LoggerWrapper, MultiObsWrapper, NormalizeObservation
import numpy as np


def get_normaliser(env_name):
    env_type = env_name.split("-")[1]
    if env_type == "datacenter":
        return RANGES_DATACENTER
    elif env_type == "5Zone":
        return RANGES_5ZONE
    elif env_type == "warehouse":
        return RANGES_WAREHOUSE
    elif env_type == "office":
        return RANGES_OFFICE
    elif env_type == "officegrid":
        return RANGES_OFFICEGRID
    elif env_type == "shop":
        return RANGES_SHOP
    else:
        raise NameError(
            'Normalization cant be use on environment :"{}", check environment name or disable normalization'.format(
                env_name
            )
        )


def make_env(env_name, gamma=None, train=False):
    def thunk():
        if "5Zone" in env_name:
            # Change range of heating and cooling setpoint to match user comfort.
            new_action_space = gym.spaces.Box(
                low=np.array([15.0, 23], dtype=np.float32),
                high=np.array([23, 30.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
            env = gym.make(
                env_name, config_params={"runperiod": (1, 1, 2022, 31, 12, 2022)}, action_space=new_action_space
            )
        else:
            env = gym.make(
                env_name, config_params={"runperiod": (1, 1, 2022, 31, 12, 2022)}
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)  # Keep to avoid base 10: '\n' error.
        env = NormalizeObservation(env, ranges=get_normaliser(env_name))
        if train:
            # Only normalise rewards when training.
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return env

    return thunk


def make_sac_env(env_name, seed=None):
    def thunk():
        if "5Zone" in env_name:
            # Change range of heating and cooling setpoint to match user comfort.
            new_action_space = gym.spaces.Box(
                low=np.array([15.0, 23], dtype=np.float32),
                high=np.array([23, 30.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
            env = gym.make(
                env_name, config_params={"runperiod": (1, 1, 2022, 31, 12, 2022)}, action_space=new_action_space
            )
        else:
            env = gym.make(
                env_name, config_params={"runperiod": (1, 1, 2022, 31, 12, 2022)}
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env = NormalizeObservation(env, ranges=get_normaliser(env_name))
        return env

    return thunk
