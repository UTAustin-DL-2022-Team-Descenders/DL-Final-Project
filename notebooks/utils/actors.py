import pystk
import torch
from torch.distributions import Bernoulli, Normal
from utils.track import state_features

def new_action_net():
    return torch.nn.Sequential(
        #torch.nn.BatchNorm1d(3*5*3),
        torch.nn.Linear(3*5*3, 20, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1, bias=False),
        torch.nn.Sigmoid()
    )

class Actor:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()
    
    def __call__(self, track_info, kart_info, **kwargs):
        f = state_features(track_info, kart_info)
        output = self.action_net(torch.as_tensor(f).view(1,-1))[0]

        action = pystk.Action()
        action.acceleration = 1.0
        steer_dist = Bernoulli(probs=output)
        action.steer = steer_dist.sample()*2-1
        return action

class GreedyActor:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()
    
    def __call__(self, track_info, kart_info, **kwargs):
        f = state_features(track_info, kart_info)
        output = self.action_net(torch.as_tensor(f).view(1,-1))[0]

        action = pystk.Action()
        action.acceleration = 1.0
        action.steer = output[0] * 2 - 1
        return action