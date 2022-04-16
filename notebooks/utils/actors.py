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
class SteeringActor:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()
    
    def __call__(self, action, f, train=False, **kwargs):        
        output = self.action_net(f)[0]        
        if train:            
            steer_dist = Bernoulli(probs=output)
            action.steer = steer_dist.sample() * 2 - 1
        else:
            action.steer = output[0] * 2 - 1
        return action

class DriftActor:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()
    
    def __call__(self, action, f, train=True, **kwargs):        
        output = self.action_net(f)[0]        
        if train:
            drift_dist = Bernoulli(probs=output[0])
            action.drift = drift_dist.sample()
        else:
            action.drift = output[0]
        return action

class Actor:
    def __init__(self, *args):
        self.nets = args
    
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
