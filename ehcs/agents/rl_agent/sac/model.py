import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import numpy as np

from ehcs.agents.rl_agent.common.model import FeatureNetwork


class SoftQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape, inter_layers=None):
        super().__init__()
        self.obs_shape = np.array(observation_shape).prod()
        self.act_shape = np.array(action_shape).prod()
        self.inter_layers_list = [64] if inter_layers is None else inter_layers
        self.feature_network = FeatureNetwork(self.inter_layers_list)
        self.train_feature_net = True
        self._build()

    def _build(self, keep_feature_network=False):
        self.input_layer = nn.Sequential(
            nn.Linear(self.obs_shape + self.act_shape, self.inter_layers_list[0]),
            nn.ReLU(),
        )
        prev_out_shape = self.inter_layers_list[-1]
        if not keep_feature_network:
            prev_out_shape = self.feature_network.build()
        self.output_layer = nn.Linear(prev_out_shape, 1)

    def build_with_features(self, obs_shape, act_shape):
        self.obs_shape = np.array(obs_shape).prod()
        self.act_shape = np.array(act_shape).prod()
        self._build(keep_feature_network=True)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.input_layer(x)
        x = self.feature_network(x)
        x = self.output_layer(x)
        return x


class Actor(nn.Module):
    def __init__(
        self, observation_shape, action_shape, action_bounds, inter_layers=None
    ):
        super().__init__()
        self.obs_shape = np.array(observation_shape).prod()
        self.act_shape = np.array(action_shape).prod()
        self.act_high, self.act_low = action_bounds
        self.inter_layers_list = [64] if inter_layers is None else inter_layers
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.feature_network = FeatureNetwork(self.inter_layers_list)
        self.train_feature_net = True
        self._build()

    def _build(self, keep_feature_network=False):
        self.input_layer = nn.Sequential(
            nn.Linear(self.obs_shape, self.inter_layers_list[0]), nn.ReLU()
        )
        prev_out_shape = self.inter_layers_list[-1]
        if not keep_feature_network:
            prev_out_shape = self.feature_network.build()
        self.fc_mean = nn.Linear(prev_out_shape, self.act_shape)
        self.fc_logstd = nn.Linear(prev_out_shape, self.act_shape)
        # Action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((self.act_high - self.act_low) / 2, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((self.act_high + self.act_low) / 2, dtype=torch.float32),
        )

    def build_with_features(self, obs_shape, act_shape, act_bounds):
        self.obs_shape = np.array(obs_shape).prod()
        self.act_shape = np.array(act_shape).prod()
        self.act_high, self.act_low = act_bounds
        self._build(keep_feature_network=True)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.feature_network(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean