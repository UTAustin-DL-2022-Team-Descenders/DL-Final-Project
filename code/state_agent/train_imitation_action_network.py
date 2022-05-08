import torch
import torch.nn.functional as F
from os import path

INPUT_CHANNELS = 19
OUTPUT_CHANNELS = 3


class ActionNetwork(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()
            self.network = torch.nn.Linear(n_input, n_output)

        def forward(self, x):
            return F.relu(self.network(x))

    def __init__(self, input_channels=INPUT_CHANNELS, hidden_layer_channels=[64, 128], output_channels=5):
        super().__init__()

        layers = []
        for layer_channels in hidden_layer_channels:
            layers.append(ActionNetwork.Block(input_channels, layer_channels))
            input_channels = layer_channels

        layers.append(torch.nn.Linear(input_channels, output_channels))

        self.network = torch.nn.Sequential(*layers)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x_std = (x - x.min()) / (x.max() - x.min())
        x_scaled = x_std * (1 - (-1)) + (-1)

        x = self.network(x_scaled)
        x = self.sigmoid(x)
        # [acceleration, steer left confidence, straight confidence, steer right confidence, brake]
        return x


class ActionNetwork1(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()
            self.network = torch.nn.Linear(n_input, n_output)

        def forward(self, x):
            return F.relu(self.network(x))

    def __init__(self, input_channels=INPUT_CHANNELS, hidden_layer_channels=[64, 128], output_channels=OUTPUT_CHANNELS, training=False):
        super().__init__()
        self.training = training

        layers = []
        for layer_channels in hidden_layer_channels:
            layers.append(ActionNetwork.Block(input_channels, layer_channels))
            input_channels = layer_channels

        # layers.append(torch.nn.Linear(input_channels, output_channels))

        self.network = torch.nn.Sequential(*layers)
        self.linear_acc = torch.nn.Linear(input_channels, 1)
        self.linear_steer = torch.nn.Linear(input_channels, 3)
        self.linear_brake = torch.nn.Linear(input_channels, 2)
        self.sigmoid_acc = torch.nn.Sigmoid()
        self.tanh_steer = torch.nn.Tanh()
        self.sigmoid_brake = torch.nn.Sigmoid()
        self.threshold_brake = torch.nn.Threshold(threshold=0.5, value=0.0)

    def forward(self, x):
        x_std = (x - x.min()) / (x.max() - x.min())
        x_scaled = x_std * (1 - (-1)) + (-1)

        x_main_net_output = self.network(x_scaled)

        acc = self.linear_acc(x_main_net_output)
        steer = self.linear_steer(x_main_net_output)
        brake = self.linear_brake(x_main_net_output)

        acc = self.sigmoid_acc(acc)
        steer = self.tanh_steer(steer)
        brake = self.sigmoid_brake(brake)
        brake = self.threshold_brake(brake)
        if self.training:
            x_out = torch.cat((acc, steer, brake), 1)
        else:
            x_out = torch.cat((acc, steer, brake), 0)
        return x_out


class ActionNetwork2(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()
            self.network = torch.nn.Linear(n_input, n_output)

        def forward(self, x):
            return F.relu(self.network(x))

    def __init__(self, input_channels=INPUT_CHANNELS, hidden_layer_channels=[128], output_channels=OUTPUT_CHANNELS):
        super().__init__()

        layers = []
        for layer_channels in hidden_layer_channels:
            layers.append(ActionNetwork.Block(input_channels, layer_channels))
            input_channels = layer_channels

        layers.append(torch.nn.Linear(input_channels, output_channels))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        x_std = (x - x.min()) / (x.max() - x.min())
        x_scaled = x_std * (1 - (-1)) + (-1)
        x = self.network(x_scaled)
        return x
