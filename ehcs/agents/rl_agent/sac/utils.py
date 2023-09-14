import gymnasium as gym
import torch
from termcolor import colored

from .model import SoftQNetwork, Actor


def create_networks(inter_layers, **kwargs):
    """
    Create a agent either using a pre-trained model or from scratch.

    Args:
        inter_layers: Intermediate layers structure.
        **path: Pre-trained model path.
        **prev_env: Gymnasium name of environment that the previous model was trained on.
        **test_env: If testing, use this environment to load trained weights.
        **curr_env: Current gymnasium environment instance.

    Returns:
        Deep neural network agent.
    """
    if "path" in kwargs:
        # Load previous model based on different conditions.
        # curr_env: Load model with no changes.
        # prev_env: Load model with saved feature network but different input and output layers.
        # test_env: Load model with no changes for evaluation.
        if "curr_env" in kwargs and "prev_env" not in kwargs:
            # Load pre-trained model without changes.
            print(colored("Model path detected!", "light_blue"))
            print(colored("Loading saved models...", "light_blue"))

            # Create agent and load based on shape info.
            obs_space_shape = kwargs["curr_env"].single_observation_space.shape
            act_space_shape = kwargs["curr_env"].single_action_space.shape
            action_high, action_low = (
                kwargs["curr_env"].single_action_space.high,
                kwargs["curr_env"].single_action_space.low,
            )

            # Create agent and load based on shape info.
            actor = Actor(
                obs_space_shape,
                act_space_shape,
                (action_high, action_low),
                inter_layers,
            )
            qf1 = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
            qf2 = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
            actor.load_state_dict(torch.load(kwargs["path"] + "/actor.pt"))
            qf1.load_state_dict(torch.load(kwargs["path"] + "/qf1.pt"))
            qf2.load_state_dict(torch.load(kwargs["path"] + "/qf2.pt"))

            # Prepare target networks
            qf1_target = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
            qf2_target = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
            qf1_target.load_state_dict(qf1.state_dict())
            qf2_target.load_state_dict(qf2.state_dict())

            print(colored("Models loaded!", "light_blue"))
            return actor, qf1, qf2, qf1_target, qf2_target
        elif "prev_env" in kwargs:
            # If fine-tuning on a different environment.
            print(colored("Model path detected!", "light_blue"))
            print(colored("Loading saved models...", "light_blue"))

            # Create env to access data.
            env = gym.make(kwargs["prev_env"])
            obs_space_shape = env.observation_space.shape
            act_space_shape = env.action_space.shape
            action_high, action_low = env.action_space.high, env.action_space.low
            env.close()

            # Create agent and load based on shape info.
            actor = Actor(
                obs_space_shape,
                act_space_shape,
                (action_high, action_low),
                inter_layers,
            )
            qf1 = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
            qf2 = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
            actor.load_state_dict(torch.load(kwargs["path"] + "/actor.pt"))
            qf1.load_state_dict(torch.load(kwargs["path"] + "/qf1.pt"))
            qf2.load_state_dict(torch.load(kwargs["path"] + "/qf2.pt"))

            # Build model while preserving feature network.
            curr_obs_shape, curr_act_shape = (
                kwargs["curr_env"].single_observation_space.shape,
                kwargs["curr_env"].single_action_space.shape,
            )
            action_high, action_low = (
            kwargs["curr_env"].single_action_space.high,
            kwargs["curr_env"].single_action_space.low,
        )
            actor.build_with_features(
                curr_obs_shape, curr_act_shape, (action_high, action_low)
            )
            qf1.build_with_features(
                curr_obs_shape,
                curr_act_shape,
            )
            qf2.build_with_features(
                curr_obs_shape,
                curr_act_shape,
            )

            # Prepare target networks
            qf1_target = SoftQNetwork(curr_obs_shape, curr_act_shape, inter_layers)
            qf2_target = SoftQNetwork(curr_obs_shape, curr_act_shape, inter_layers)
            qf1_target.load_state_dict(qf1.state_dict())
            qf2_target.load_state_dict(qf2.state_dict())

            print(colored("Models loaded!", "light_blue"))
            return actor, qf1, qf2, qf1_target, qf2_target
        elif "test_env" in kwargs:
            # If testing, load pre-trained model without changes.
            print(colored("Model path detected!", "light_blue"))
            print(colored("Loading saved model...", "light_blue"))

            # Create agent and load based on shape info.
            obs_space_shape = kwargs["test_env"].observation_space.shape
            act_space_shape = kwargs["test_env"].action_space.shape
            action_high, action_low = (
                kwargs["test_env"].action_space.high,
                kwargs["test_env"].action_space.low,
            )
            actor = Actor(
                obs_space_shape,
                act_space_shape,
                (action_high, action_low),
                inter_layers,
            )
            actor.load_state_dict(torch.load(kwargs["path"] + "/actor.pt"))
            actor.eval()
            print(colored("Model loaded!", "light_blue"))
            return actor
    else:
        # If no pre-trained model will be used, create a new model.
        print(colored("No model path detected!", "light_blue"))
        print(colored("Creating new model...", "light_blue"))

        # Create agent and load based on shape info.
        obs_space_shape = kwargs["curr_env"].single_observation_space.shape
        act_space_shape = kwargs["curr_env"].single_action_space.shape
        action_high, action_low = (
            kwargs["curr_env"].single_action_space.high,
            kwargs["curr_env"].single_action_space.low,
        )
        actor = Actor(
            obs_space_shape,
            act_space_shape,
            (action_high, action_low),
            inter_layers,
        )
        qf1 = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
        qf2 = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)

        # Prepare target networks
        qf1_target = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
        qf2_target = SoftQNetwork(obs_space_shape, act_space_shape, inter_layers)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        print(colored("Model created!", "light_blue"))
        return actor, qf1, qf2, qf1_target, qf2_target
