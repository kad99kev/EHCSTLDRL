import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureNetwork(nn.Module):
    def __init__(self, layers_list):
        super().__init__()
        self.layers_list = layers_list

    def build(self):
        prev_out_shape = self.layers_list[0]
        self.network = nn.Sequential()
        for layer in self.layers_list:
            if layer == "dropout":
                self.network.append(nn.Dropout(0.2))
                continue
            self.network.append(nn.Linear(prev_out_shape, layer))
            self.network.append(nn.ReLU())
            prev_out_shape = layer
        return prev_out_shape

    def forward(self, x):
        return self.network(x)
