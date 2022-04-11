from argparse import Action
import torch
import torch.nn.functional as F

# Assumes extract_features returns 28 channels
INPUT_CHANNELS = 28

# Assumes network outputs 6 different actions
# Output channel order: [acceleration, steer, brake, drift, fire, nitro]
# Boolean outputs will be thresholded by 0.5
OUTPUT_CHANNELS = 6

# TODO: Currently a barebones network. Fix me!
class ActionNetwork(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()
            self.network = torch.nn.Linear(n_input, n_output)

        def forward(self, x):
            return F.relu(self.network(x))

    def __init__(self, input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS):
        super().__init__()

        self.network = ActionNetwork.Block(input_channels, output_channels)


    def forward(self, x):
        return self.network(x)


def save_model(model):
    from os import path
    model_scripted = torch.jit.script(model)
    model_scripted.save(path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt'))


def load_model():
    from os import path
    model = torch.jit.load( path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt'))
    return model