import numpy as np
import torch


import random
import gymnasium as gym
import wandb
import os
from termcolor import colored
import torch.optim as optim
import pathlib
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import time

from .utils import create_networks

from ehcs.config import parse_config
from ehcs.utils import make_sac_env
from ehcs.utils.loggers import TrainLogger


def run(config_path):
    # Configuration setup
    cfg = parse_config(config_path)

    train_params = cfg["train"]

    env_names = train_params["name"]

    # Hyperparameter setup.
    agent_hyperparams = cfg["agent"]["hyperparams"]
    agent_hyperparams["num_envs"] = len(env_names)
    CHECKPOINT_FREQ = cfg["agent"]["checkpoint_freq"]
    agent_hyperparams["buffer_size"] = int(float(agent_hyperparams["buffer_size"]))
    agent_hyperparams["learning_starts"] = int(
        float(agent_hyperparams["learning_starts"])
    )
    agent_hyperparams["policy_lr"] = float(agent_hyperparams["policy_lr"])
    agent_hyperparams["q_lr"] = float(agent_hyperparams["q_lr"])

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = cfg["torch_deterministic"]

    # Environment setup
    sub_envs = [
        make_sac_env(env_name, seed=cfg["seed"]) for i, env_name in enumerate(env_names)
    ]
    envs = gym.vector.SyncVectorEnv(sub_envs)
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # WandB setup
    save_path = pathlib.Path(config_path).parents[0] / "train"
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
            name=f"train_{cfg['run_name']}",
            dir=wandb_config["save_path"],
            sync_tensorboard=True,
            config=dict(
                **agent_hyperparams,
                **{
                    "seed": cfg["seed"],
                    "cuda": cfg["cuda"],
                    "mode": "train",
                    "save_freq": CHECKPOINT_FREQ,
                    "layers": cfg["agent"]["layers"],
                    "env_name": "-".join(env_names[0].split("-")[1:4]),
                    "algorithm": cfg["agent"]["algorithm"],
                    "method": cfg["method"],
                    "reward": cfg["reward"]
                },
            ),
        )
        writer = SummaryWriter(f"{save_path}/tensorboard/")

        # Initialise logger.
        obs_variables, action_variables = envs.call("variables")[0].values()
        train_logger = TrainLogger(env_names, obs_variables, action_variables, writer)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["cuda"] else "cpu"
    )
    print(colored(f"{device} is being utilised.", "light_cyan"))

    # Networks setup
    if "model" in train_params:
        # Load previously saved networks.
        path = train_params["model"]["path"]
        if "env_trained" in train_params["model"]:
            # If previous environment is different from current environment,
            # reset input/output layers.
            env = train_params["model"]["env_trained"]
            actor, qf1, qf2, qf1_target, qf2_target = create_networks(
                inter_layers=cfg["agent"]["layers"],
                prev_env=env,
                curr_env=envs,
                path=path,
            )
        else:
            actor, qf1, qf2, qf1_target, qf2_target = create_networks(
                inter_layers=cfg["agent"]["layers"], curr_env=envs, path=path
            )
    else:
        # Create new networks.
        actor, qf1, qf2, qf1_target, qf2_target = create_networks(
            inter_layers=cfg["agent"]["layers"], curr_env=envs
        )

    # Change device.
    actor.to(device)
    qf1.to(device)
    qf2.to(device)
    qf1_target.to(device)
    qf2_target.to(device)

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=agent_hyperparams["q_lr"]
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()), lr=agent_hyperparams["policy_lr"]
    )

    # Automatic entropy tuning
    if agent_hyperparams["autotune"]:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=agent_hyperparams["q_lr"])
    else:
        alpha = agent_hyperparams["alpha"]

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        agent_hyperparams["buffer_size"],
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=len(env_names),
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=cfg["seed"])
    for global_step in range(agent_hyperparams["total_timesteps"]):
        # ALGO LOGIC: put action logic here
        if global_step < agent_hyperparams["learning_starts"]:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        if wandb_config:
            train_logger.log_update(
                envs.call("get_unwrapped_obs"), terminated, infos, global_step
            )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # Only print when at least 1 env is done
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                print(
                    colored(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}",
                        "light_magenta",
                    )
                )
                if wandb_config:
                    writer.add_scalar(
                        f"{env_names[i]}/episode/episodic_return",
                        info["episode"]["r"],
                        global_step,
                    )
                    writer.add_scalar(
                        f"{env_names[i]}/episode/episodic_length",
                        info["episode"]["l"],
                        global_step,
                    )
                    # Log episode.
                    train_logger.log_episode(i, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(terminated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        # print(obs, real_next_obs, actions.shape, rewards, terminated, infos)
        rb.add(obs, real_next_obs, actions, rewards, terminated, [infos])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > agent_hyperparams["learning_starts"]:
            data = rb.sample(agent_hyperparams["batch_size"])
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * agent_hyperparams["gamma"] * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if (
                global_step % agent_hyperparams["policy_frequency"] == 0
            ):  # TD 3 Delayed update support
                for _ in range(
                    agent_hyperparams["policy_frequency"]
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if agent_hyperparams["autotune"]:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % agent_hyperparams["target_network_frequency"] == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        agent_hyperparams["tau"] * param.data
                        + (1 - agent_hyperparams["tau"]) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        agent_hyperparams["tau"] * param.data
                        + (1 - agent_hyperparams["tau"]) * target_param.data
                    )

            if global_step % 100 == 0:
                if wandb_config:
                    writer.add_scalar(
                        "losses/qf1_values", qf1_a_values.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "losses/qf2_values", qf2_a_values.mean().item(), global_step
                    )
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step
                    )
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    writer.add_scalar(
                        "time/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    writer.add_scalar(
                        "time/time_elapsed", time.time() - start_time, global_step
                    )
                    if agent_hyperparams["autotune"]:
                        writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )
                print(
                    colored(
                        f"SPS: {int(global_step / (time.time() - start_time))} | Steps: {global_step} / {agent_hyperparams['total_timesteps']}",
                        "light_magenta",
                    )
                )

        # Model saving.
        if wandb_config:
            if global_step % CHECKPOINT_FREQ == 0 or global_step == (
                agent_hyperparams["total_timesteps"] - 1
            ):
                print(colored("Saving model...", "light_blue"))
                # Create folder for model saving.
                model_save_path = pathlib.Path(
                    f"{save_path}/checkpoints/agent_{global_step}/"
                )
                model_save_path.mkdir(parents=True, exist_ok=True)
                models = {
                    "actor": actor,
                    "qf1": qf1,
                    "qf2": qf2,
                }
                for model_name, model in models.items():
                    torch.save(model.state_dict(), f"{model_save_path}/{model_name}.pt")
                    wandb.save(
                        f"{model_save_path}/{model_name}.pt",
                        base_path=model_save_path.parent,
                        policy="now",
                    )
                print(colored("Models saved!", "light_blue"))

    envs.close()
    writer.close()
