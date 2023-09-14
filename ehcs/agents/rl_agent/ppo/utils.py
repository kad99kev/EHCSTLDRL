import gymnasium as gym
import torch
from termcolor import colored

from .model import Agent


def create_agent(inter_layers, **kwargs):
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
            # (Only works for environment with same observation and action spaces).
            print(colored("Model path detected!", "light_blue"))
            print(colored("Loading saved model...", "light_blue"))

            # Create agent and load based on shape info.
            agent = Agent(
                kwargs["curr_env"].observation_space.shape,
                kwargs["curr_env"].action_space.shape,
                inter_layers,
            )
            agent.load_state_dict(torch.load(kwargs["path"]))

            print(colored("Model loaded!", "light_blue"))
            return agent
        elif "prev_env" in kwargs:
            # If fine-tuning on a different environment.
            print(colored("Model path detected!", "light_blue"))
            print(colored("Loading saved model...", "light_blue"))

            # Create env to access data.
            env = gym.make(kwargs["prev_env"])
            obs_space_shape = env.observation_space.shape
            act_space_shape = env.action_space.shape
            env.close()

            # Create agent and load based on shape info.
            agent = Agent(obs_space_shape, act_space_shape, inter_layers=inter_layers)
            agent.load_state_dict(torch.load(kwargs["path"]))

            # Build model while preserving feature network.
            agent.build_with_features(
                kwargs["curr_env"].single_observation_space.shape,
                kwargs["curr_env"].single_action_space.shape,
            )
            print(colored("Model loaded!", "light_blue"))
            return agent
        elif "test_env" in kwargs:
            # If testing, load pre-trained model without changes.
            print(colored("Model path detected!", "light_blue"))
            print(colored("Loading saved model...", "light_blue"))

            # Create agent and load based on shape info.
            agent = Agent(
                kwargs["test_env"].observation_space.shape,
                kwargs["test_env"].action_space.shape,
                inter_layers,
            )
            agent.load_state_dict(torch.load(kwargs["path"]))
            agent.eval()
            print(colored("Model loaded!", "light_blue"))
            return agent
    else:
        # If no pre-trained model will be used, create a new model.
        print(colored("No model path detected!", "light_blue"))
        print(colored("Creating new model...", "light_blue"))

        agent = Agent(
            kwargs["curr_env"].single_observation_space.shape,
            kwargs["curr_env"].single_action_space.shape,
            inter_layers,
        )
        print(colored("Model created!", "light_blue"))
        return agent
