import os
import pathlib
import random
import time

import gymnasium as gym
import numpy as np
import torch
import wandb
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from ehcs.config import parse_config
from ehcs.utils import make_env
from ehcs.utils.loggers import EvalLogger

from .utils import create_networks


def run(config_path):
    cfg = parse_config(config_path)

    test_params = cfg["test"]
    inter_layers = cfg["agent"]["layers"]

    # Initialise the environment.
    env = make_env(test_params["name"])()

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = cfg["torch_deterministic"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["cuda"] else "cpu"
    )
    print(colored(f"{device} is being utilised.", "light_cyan"))

    # Initialise actor.
    actor = create_networks(
        inter_layers=inter_layers, path=test_params["path"], test_env=env
    )
    actor.to(device)
    print(colored(f"Model in train mode?: {actor.training}.", "light_cyan"))

    # Logging setup.
    save_path = pathlib.Path(config_path).parents[0] / "test"
    if "wandb" in cfg:
        wandb_config = cfg["wandb"]
        wandb_config["save_path"] = save_path
    else:
        wandb_config = None

    # wandb_config = None
    if wandb_config:
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=f"test_{cfg['run_name']}",
            dir=wandb_config["save_path"],
            sync_tensorboard=True,
            config={
                "seed": cfg["seed"],
                "mode": "test",
                "episodes": test_params["episodes"],
                "log_interval": test_params["log_interval"],
                "layers": cfg["agent"]["layers"],
                "env_name": "-".join(test_params["name"].split("-")[1:4]),
            },
        )
        writer = SummaryWriter(f"{save_path}/tensorboard/")

        # Setup evaluation logger.
        obs_variables, action_variables = env.variables.values()
        eval_logger = EvalLogger(obs_variables, action_variables, writer)

    # Start controller.
    global_step = 0
    start_time = time.time()
    print(
        colored(f"{test_params['episodes']} episodes will be executed.", "light_cyan")
    )

    for episode in range(1, test_params["episodes"] + 1):
        print(
            colored(
                f"Running episode {episode} of {test_params['episodes']}.", "light_cyan"
            )
        )
        next_obs, _ = env.reset(seed=cfg["seed"])
        next_obs = torch.Tensor(next_obs).reshape(1, -1).to(device)
        terminated = False
        while not terminated:
            global_step += 1

            with torch.no_grad():
                action, _, _ = actor.get_action(next_obs)

            next_obs, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()[0]
            )

            # Logging.
            if wandb_config:
                # Log step values.
                if global_step % test_params["log_interval"] == 0:
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

            # Convert next obs to a tensor.
            next_obs = torch.Tensor(next_obs).reshape(1, -1).to(device)

        print(
            colored(
                f"SPS: {int(global_step / (time.time() - start_time))}", "light_magenta"
            )
        )

    env.close()
    writer.close()
