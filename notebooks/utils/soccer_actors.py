import torch
from utils.track import SoccerFeatures
from utils.base_actors import BaseActor, LinearWithTanh, LinearWithSigmoid, Agent as BaseAgent

class SteeringActor(BaseActor):
    
    def __init__(self, action_net=None, train=None, **kwargs):
        # Steering action_net
        # inputs: delta steering angle
        super().__init__(LinearWithTanh(1, 1) if action_net is None else action_net, train=train)

    def __call__(self, action, f, train=False, **kwargs):        
        output = self.action_net(f)
        if self.train is not None:
            train = self.train
        if train:                        
            action.steer = self.sample_bernoulli(output) * 2 - 1
        else:
            action.steer = output[0] # raw output
        return action

    def reward(self, action, extractor, selected_features_curr, selected_features_next):
        [current_angle] = selected_features_curr
        [next_angle] = selected_features_next
        return steering_angle_reward(current_angle, next_angle)        
    
    def extract_greedy_action(self, action):
        return action.steer > 0

    def select_features(self, features, features_vec):
        delta_steering_angle = features.select_delta_steering(features_vec)        
        return torch.Tensor([
            delta_steering_angle
        ])

class DriftActor(BaseActor):
    
    def __init__(self, action_net=None, train=None, **kwargs):
        # Steering action_net
        # inputs: delta steering angle, delta lateral distance        
        super().__init__(LinearWithSigmoid(2, 1) if action_net is None else action_net, train=train)

    def __call__(self, action, f, train=True, **kwargs):        
        output = self.action_net(f)
        if self.train is not None:
            train = self.train
        if train:            
            action.drift = self.sample_bernoulli(output) > 0.5
        else:
            # drift is a binary value
            action.drift = output[0] > 0.5
        
        return action

    def reward(self, action, extractor, selected_features_curr, selected_features_next):
        [current_angle] = selected_features_curr
        [next_angle] = selected_features_next        
        return steering_angle_reward(current_angle, next_angle)
        
    def extract_greedy_action(self, action):
        return action.drift > 0.5

    def select_features(self, features, features_vec):
        delta_steering_angle = features.select_delta_steering(features_vec)        
        return torch.tensor([
            delta_steering_angle
        ])

class Agent(BaseAgent):
    def __init__(self, *args, target_speed=10.0, **kwargs):
        super().__init__(*args, extractor=SoccerFeatures(), target_speed=target_speed, **kwargs)
