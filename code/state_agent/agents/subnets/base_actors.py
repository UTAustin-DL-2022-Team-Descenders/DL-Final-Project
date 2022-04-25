# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import torch
from torch.distributions import Bernoulli, Normal

def new_action_net(n_outputs=1, type="linear_tanh"):
    if type == "linear_sigmoid":
        return LinearWithSigmoid(n_outputs)
    elif type == "linear_tanh":
        return LinearWithTanh(n_outputs)
    else:
        raise Exception("Unknown action net")
        
class BaseNetwork(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()        

    def forward(self, x, train=None):
        return self.net(x)

class LinearNetwork(BaseNetwork):
    
    def __init__(self, activation, n_inputs, n_outputs, n_hidden, bias) -> None:
        super().__init__()
        self.activation = activation.__name__
        self.net = torch.nn.Sequential(
            #torch.nn.BatchNorm1d(3*5*3),
            torch.nn.Linear(n_inputs, n_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_outputs, bias=bias),
            activation()
            # torch.nn.HardSigmoid() # can train better than sigmoid, with less optimal output
        )

class LinearWithSigmoid(LinearNetwork):

    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=20, bias=False) -> None:
        super().__init__(torch.nn.Sigmoid, n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, bias=bias)

    def forward(self, x):
        return self.net(x)
    
class LinearWithTanh(LinearNetwork):

    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=20, bias=False) -> None:        
        super().__init__(torch.nn.Tanh, n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, bias=bias)


    def forward(self, x):
        if self.training:
            output = self.net(x)
            # the training output needs to be a probability
            output = (output + 1) / 2
            return output
        else:
            return self.net(x)

class BaseActor:

    def __init__(self, action_net, train=None, reward_type=None):
        self.action_net = action_net.cpu().eval()
        self.train = train
        self.reward_type = reward_type

    def copy(self, action_net):
        return self.__class__(action_net, train=self.train, reward_type=self.reward_type)

    def sample_bernoulli(self, output):
        if self.action_net.activation == "Tanh" or \
           self.action_net.activation == "Hardtanh":
            output = (output + 1) / 2
        output = Bernoulli(probs=output).sample()
        return output

    def select_features(self, state_features):

        # this is only called for top level actors; nested actors are given features directly from their ancestors
        pass

