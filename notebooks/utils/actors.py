from ast import In
from numpy import Inf
import pystk
import torch
import numpy as np
from torch.distributions import Bernoulli, Normal
from utils.track import state_features
from utils.rewards import lateral_distance_reward, lateral_distance_causal_reward, distance_traveled_reward

def new_action_net():
    return torch.nn.Sequential(
        #torch.nn.BatchNorm1d(3*5*3),
        torch.nn.Linear(3*5*3, 20, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1, bias=True),
        torch.nn.Sigmoid()
    )

class BaseActor:

    def __init__(self, action_net, train=None):
        self.action_net = action_net.cpu().eval()
        self.train = train
    
class SteeringActor(BaseActor):
    
    def __call__(self, action, f, train=False, **kwargs):        
        output = self.action_net(f)[0]
        if self.train is not None:
            train = self.train
        if train:            
            steer_dist = Bernoulli(probs=output)
            action.steer = steer_dist.sample() * 2 - 1
        else:
            action.steer = output[0] * 2 - 1
        return action

    def reward(self, action, current_lat=Inf, next_lat=Inf, **kwargs):
        return lateral_distance_reward(current_lat, next_lat)
    
    def extract_greedy_action(self, action):
        return action.steer > 0

class DriftActor(BaseActor):
    
    def __call__(self, action, f, train=True, **kwargs):        
        output = self.action_net(f)[0] 
        if self.train is not None:
            train = self.train
        if train:
            drift_dist = Bernoulli(probs=output)
            action.drift = drift_dist.sample() > 0.5
        else:
            # drift is a binary value
            action.drift = output[0] > 0.5
        
        return action

    def reward(self, action, current_distance=Inf, next_distance=Inf, current_lat=Inf, next_lat=Inf, **kwargs): 
        return lateral_distance_causal_reward(current_lat, next_lat)         

    def extract_greedy_action(self, action):
        return action.drift > 0.5

class Actor:
    def __init__(self, *args):
        self.nets = args
        self.last_output = torch.Tensor([0, 0, 0, 0, 0])
    
    def invoke_nets(self, action, f):
        for net in self.nets:
            net(action, f, train=True)        

    def __call__(self, track_info, kart_info, **kwargs):
        action = pystk.Action()        
        action.acceleration = 1.0
        f = state_features(track_info, kart_info)
        f = torch.as_tensor(f).view(1,-1)
        self.invoke_nets(action, f)        
        return action

class GreedyActor(Actor):
    
    def invoke_nets(self, action, f):
        for net in self.nets:
            net(action, f, train=False)
