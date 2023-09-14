from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from ehcs.agents.rl_agent.common.model import FeatureNetwork, layer_init


class Critic(nn.Module):
    def __init__(self, observation_shape, inter_layers=None):
        super().__init__()
        self.obs_shape = observation_shape
        self.inter_layers_list = [64] if inter_layers is None else inter_layers
        self.feature_network = FeatureNetwork(self.inter_layers_list)
        self._build()

    def __str__(self):
        return f"{str(self.input_layer)}\n{str(self.feature_network)}\n{str(self.output_layer)}"

    def _build(self, keep_feature_network=False):
        self.input_layer = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, self.inter_layers_list[0])), nn.ReLU()
        )
        prev_out_shape = self.inter_layers_list[-1]
        if not keep_feature_network:
            prev_out_shape = self.feature_network.build()
        self.output_layer = layer_init(nn.Linear(prev_out_shape, 1), std=1.0)

    def build_with_features(self, obs_shape):
        self.obs_shape = obs_shape
        self._build(keep_feature_network=True)

    def forward(self, inputs):
        inp_outs = self.input_layer(inputs)
        feature_net_outs = self.feature_network(inp_outs)
        outs = self.output_layer(feature_net_outs)
        return outs


class Actor(nn.Module):
    def __init__(self, observation_shape, action_shape, inter_layers=None):
        super().__init__()
        self.obs_shape = observation_shape
        self.act_shape = action_shape
        self.inter_layers_list = [64] if inter_layers is None else inter_layers
        self.feature_network = FeatureNetwork(self.inter_layers_list)
        self._build()
        self.step = 0

    def __str__(self):
        return f"{str(self.input_layer)}\n{str(self.feature_network)}\n{str(self.output_layer)}"

    def _build(self, keep_feature_network=False):
        self.input_layer = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, self.inter_layers_list[0])), nn.ReLU()
        )
        prev_out_shape = self.inter_layers_list[-1]
        if not keep_feature_network:
            prev_out_shape = self.feature_network.build()
        self.output_layer = layer_init(
            nn.Linear(prev_out_shape, self.act_shape), std=0.01
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.act_shape)))

    def build_with_features(self, obs_shape, act_shape):
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self._build(keep_feature_network=True)

    def forward(self, inputs, action=None):
        # Actor mean
        inp_outs = self.input_layer(inputs)
        inter_outs = self.feature_network(inp_outs)
        action_mean = self.output_layer(inter_outs)

        # Actor std
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class Agent(nn.Module):
    def __init__(self, observation_shape, action_shape, inter_layers=None):
        super().__init__()
        self.obs_shape = np.array(observation_shape).prod()
        self.act_shape = np.array(action_shape).prod()
        self.inter_layers_list = [64] if inter_layers is None else inter_layers
        self.train_feature_net = True
        self._build()

    def __str__(self):
        return f"Critic:\n{str(self.critic)}\n\nActor:\n{str(self.actor)}"

    def _build(self, keep_features=False):
        if keep_features:
            self.critic.build_with_features(self.obs_shape)
            self.actor.build_with_features(self.obs_shape, self.act_shape)
        else:
            self.critic = Critic(self.obs_shape, self.inter_layers_list)
            self.actor = Actor(self.obs_shape, self.act_shape, self.inter_layers_list)

    def build_with_features(self, obs_shape, act_shape):
        self.obs_shape = np.array(obs_shape).prod()
        self.act_shape = np.array(act_shape).prod()
        self._build(keep_features=True)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action, log_prob_sum, prob_sum = self.actor(x, action=action)
        critic_out = self.critic(x)
        return action, log_prob_sum, prob_sum, critic_out


if __name__ == "__main__":
    ## Saving
    obs_shape, act_shape = (20,), (5,)
    agent = Agent(obs_shape, act_shape)
    print(agent)
    torch.save(agent.state_dict(), "./trial_model.pth")

    ## Loading
    obs_shape, act_shape = (20,), (5,)
    agent = Agent(obs_shape, act_shape)
    agent.load_state_dict(torch.load("./trial_model.pth"))
    agent.build_with_features((10,), (3,))
    print(agent)
