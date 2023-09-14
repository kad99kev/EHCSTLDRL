import os
import pathlib
import time

import gymnasium as gym
import numpy as np
import sinergym.utils.controllers as controllers
from sinergym.utils.rewards import ExpReward
import wandb
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from ehcs.config import parse_config
from ehcs.utils.loggers import EvalLogger


def get_controller(env_name):
    if "5Zone" in env_name:
        return controllers.RBC5Zone
    elif "data" in env_name:
        return controllers.RBCDatacenter
    else:
        return controllers.RandomController


def run(config_path):
    cfg = parse_config(config_path)

    cntr_params = cfg["controller"]

    # Initialise environment.
    env = gym.make(
        cntr_params["name"], config_params={"runperiod": (1, 1, 2022, 31, 12, 2022)}
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Initialise controller agent.
    Controller = get_controller(cntr_params["name"])
    agent = Controller(env)

    # Logging setup.
    save_path = pathlib.Path(config_path).parents[0] / "controller"
    if "wandb" in cfg:
        wandb_config = cfg["wandb"]
        wandb_config["save_path"] = save_path
    else:
        wandb_config = None

    # wandb_config = None
    if wandb_config:
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        cfg["run_name"] += "_" + str(cfg["seed"])
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=cfg['run_name'],
            dir=wandb_config["save_path"],
            sync_tensorboard=True,
            config={
                "seed": cfg["seed"],
                "mode": "controller",
                "algorithm": "controller",
                "method": "controller",
                "reward": cfg["reward"],
                "total_timesteps": cntr_params["total_timesteps"],
                "log_interval": cntr_params["log_interval"],
                "env_name": "-".join(cntr_params["name"].split("-")[1:4]),
            },
        )
        writer = SummaryWriter(f"{save_path}/tensorboard/")

        # Setup evaluation logger.
        obs_variables, action_variables = env.variables.values()
        eval_logger = EvalLogger(obs_variables, action_variables, writer)

    # Start controller.
    start_time = time.time()
    print(
        colored(f"{cntr_params['total_timesteps']} timesteps will be executed.", "light_cyan")
    )


    next_obs, _ = env.reset(seed=cfg["seed"])
    for global_step in range(cntr_params["total_timesteps"]):
        if global_step % cntr_params["log_interval"] == 0:
            print(
            colored(
                f"Timesteps: {global_step + 1} / {cntr_params['total_timesteps']}", "light_cyan"
            )
        )
        
        action = agent.act(next_obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Logging.
        if wandb_config:
            # Log step values.
            eval_logger.log_update(next_obs, terminated, info, global_step)

            # Log episode values.
            if terminated:
                print(
                    colored(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}",
                        "light_magenta",
                    )
                )
                writer.add_scalar(
                    "episode/episodic_return",
                    info["episode"]["r"],
                    global_step,
                )
                writer.add_scalar(
                    "episode/episodic_length",
                    info["episode"]["l"],
                    global_step,
                )
                eval_logger.log_episode(global_step)

                # Reset environment.
                next_obs, _ = env.reset(seed=cfg["seed"])
        
        if global_step % cntr_params["log_interval"] == 0:
            print(
                colored(
                    f"SPS: {int(global_step / (time.time() - start_time))}", "light_magenta"
                )
            )

    env.close()
    writer.close()
