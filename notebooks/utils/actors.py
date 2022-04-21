from ast import In
from numpy import Inf
import pystk
import torch
import numpy as np
from torch.distributions import Bernoulli, Normal
from utils.track import state_features, state_features_soccer
from utils.rewards import lateral_distance_reward, lateral_distance_causal_reward, distance_traveled_reward, steering_angle_reward

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

class LinearWithSigmoid(BaseNetwork):

    def __init__(self, n_outputs=1) -> None:
        super().__init__()
        self.activation = torch.nn.Sigmoid
        self.net = torch.nn.Sequential(
            #torch.nn.BatchNorm1d(3*5*3),
            torch.nn.Linear(3*5*3, 20, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(20, n_outputs, bias=False),
            self.activation()
            # torch.nn.HardSigmoid() # can train better than sigmoid, with less optimal output
        )
    
class LinearWithTanh(BaseNetwork):

    def __init__(self, n_outputs=1) -> None:
        super().__init__()
        self.activation = torch.nn.Tanh
        self.net = torch.nn.Sequential(
            #torch.nn.BatchNorm1d(3*5*3),
            torch.nn.Linear(3*5*3, 20, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(20, n_outputs, bias=False),
            self.activation()
            # torch.nn.Hardtanh() # can train better than tanh, with less optimal output
        )

    def forward(self, x, train=None):
        if train == "reinforce":
            output = self.net(x)
            # the training output needs to be a probability
            output = (output + 1) / 2
            return output
        else:
            return super().forward(x)

class BaseActor:

    def __init__(self, action_net, train=None, reward_type="angle"):
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
    
class SteeringActor(BaseActor):
    
    def __call__(self, action, f, train=False, **kwargs):        
        output = self.action_net(f)[0]
        if self.train is not None:
            train = self.train
        if train:                        
            action.steer = self.sample_bernoulli(output) * 2 - 1
        else:
            action.steer = output[0]
        return action

    def reward(self, action, current_angle=Inf, next_angle=Inf, current_lat=Inf, next_lat=Inf, **kwargs):
        if self.reward_type == "angle":
            return steering_angle_reward(current_angle, next_angle)
        else:
            return lateral_distance_reward(current_lat, next_lat)
    
    def extract_greedy_action(self, action):
        return action.steer > 0

class DriftActor(BaseActor):
    
    def __call__(self, action, f, train=True, **kwargs):        
        output = self.action_net(f)[0] 
        if self.train is not None:
            train = self.train
        if train:            
            action.drift = self.sample_bernoulli(output) > 0.5
        else:
            # drift is a binary value
            action.drift = output[0] > 0.5
        
        return action

    def reward(self, action, current_angle=Inf, next_angle=Inf, current_lat=Inf, next_lat=Inf, **kwargs): 
        if self.reward_type == "angle":
            return steering_angle_reward(current_angle, next_angle)
        else:
            return lateral_distance_causal_reward(current_lat, next_lat)        

    def extract_greedy_action(self, action):
        return action.drift > 0.5

class Agent:
    def __init__(self, *args, **kwargs):
        self.nets = args
        self.accel = kwargs['accel'] if 'accel' in kwargs else 1.0
        self.last_output = torch.Tensor([0, 0, 0, 0, 0])        
    
    def invoke_nets(self, action, f):
        for net in self.nets:
            net(action, f, train=False)        

    def __call__(self, track_info, kart_info, soccer_state=None, **kwargs):
        action = pystk.Action()        
        action.acceleration = self.accel
        if track_info:
            f = state_features(track_info, kart_info)
        else:
            f = state_features_soccer(track_info, kart_info, soccer_state)
        f = torch.as_tensor(f).view(1,-1)
        self.invoke_nets(action, f)        
        return action

class TrainingAgent(Agent):
    
    def invoke_nets(self, action, f):
        for net in self.nets:
            net(action, f, train=True)        
