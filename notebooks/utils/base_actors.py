import torch
from torch.distributions import Bernoulli, Normal
import numpy as np
import pystk

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
        self.activation = None

    def forward(self, x, train=None):
        return self.net(x)

class LinearNetwork(BaseNetwork):
    
    def __init__(self, activation, n_inputs, n_outputs, n_hidden, bias) -> None:
        super().__init__()
        self.activation = activation
        self.net = torch.nn.Sequential(
            #torch.nn.BatchNorm1d(3*5*3),
            torch.nn.Linear(n_inputs, n_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_outputs, bias=bias),
            self.activation()
            # torch.nn.HardSigmoid() # can train better than sigmoid, with less optimal output
        )

class LinearWithSigmoid(LinearNetwork):

    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=20, bias=False) -> None:
        super().__init__(torch.nn.Sigmoid, n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, bias=bias)
    
class LinearWithTanh(LinearNetwork):

    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=20, bias=False) -> None:        
        super().__init__(torch.nn.Tanh, n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, bias=bias)

    def forward(self, x, train=None):
        if train == "reinforce":
            output = self.net(x)
            # the training output needs to be a probability
            output = (output + 1) / 2
            return output
        else:
            return super().forward(x)

class BaseActor:

    def __init__(self, action_net, train=None, reward_type=None):
        self.action_net = action_net.cpu().eval()
        self.train = train
        self.reward_type = reward_type

    def copy(self, action_net):
        return self.__class__(action_net, train=self.train, reward_type=self.reward_type)

    def sample_bernoulli(self, output):
        if self.action_net.activation == torch.nn.Tanh or \
           self.action_net.activation == torch.nn.Hardtanh:
            output = (output + 1) / 2
        output = Bernoulli(probs=output).sample()
        return output

    def select_features(self, state_features):

        # this is only called for top level actors; nested actors are given features directly from their ancestors
        pass

class Agent:
    def __init__(self, *args, train=False, **kwargs):
        self.nets = args
        self.train = train
        self.accel = kwargs['accel'] if 'accel' in kwargs else 1.0
        self.last_output = torch.Tensor([0, 0, 0, 0, 0])        
    
    def invoke_actor(self, actor, extractor, action, f):
        actor(action, actor.select_features(extractor, f), train=self.train)       

    def invoke_actors(self, extractor, action, f):
        [self.invoke_actor(actor, extractor, action, f) for actor in self.nets]
            
    def __call__(self, extractor, track_info, kart_info, soccer_state=None, **kwargs):
        action = pystk.Action()        
        action.acceleration = self.accel
        f = extractor.get_feature_vector(track_info, kart_info, soccer_state=soccer_state)        
        f = torch.as_tensor(f).view(-1)
        self.invoke_actors(extractor, action, f)        
        return action
