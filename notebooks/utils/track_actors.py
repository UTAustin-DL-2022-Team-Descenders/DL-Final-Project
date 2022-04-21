import torch
import numpy as np
from numpy import Inf
from utils.base_actors import BaseActor, LinearWithTanh, LinearWithSigmoid, Agent
from utils.track import TrackFeatures
from utils.rewards import lateral_distance_reward, lateral_distance_causal_reward, steering_angle_reward

class SteeringActor(BaseActor):
    
    def __init__(self, action_net=None, train=None, reward_type="angle"):
        # Steering action_net        
        super().__init__(LinearWithTanh(45, 1) if action_net is None else action_net, train=train, reward_type=reward_type)

    def __call__(self, action, f, train=False, **kwargs):        
        output = self.action_net(f)
        if self.train is not None:
            train = self.train
        if train:                        
            action.steer = self.sample_bernoulli(output) * 2 - 1
        else:
            action.steer = output[0] # raw output
        return action

    def reward(self, action, extractor, features_curr, features_next):
        if self.reward_type == "angle":
            current_angle = extractor.select_delta_steering(features_curr)
            next_angle = extractor.select_delta_steering(features_next)        
            return steering_angle_reward(current_angle, next_angle)
        else:
            current_lat = extractor.select_lateral_distance(features_curr)
            next_lat = extractor.select_lateral_distance(features_next)        
            return lateral_distance_reward(current_lat, next_lat)
    
    def extract_greedy_action(self, action):
        return action.steer > 0

    def select_features(self, features, features_vec):
        return features_vec

class DriftActor(BaseActor):
    
    def __init__(self, action_net=None, train=None, reward_type="angle"):
        # Steering action_net
        super().__init__(LinearWithSigmoid(45, 1) if action_net is None else action_net, train=train, reward_type=reward_type)

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

    def reward(self, action, extractor, features_curr, features_next):
        if self.reward_type == "angle":
            current_angle = extractor.select_delta_steering(features_curr)
            next_angle = extractor.select_delta_steering(features_next)        
            return steering_angle_reward(current_angle, next_angle)
        else:
            current_lat = extractor.select_lateral_distance(features_curr)
            next_lat = extractor.select_lateral_distance(features_next)        
            return lateral_distance_causal_reward(current_lat, next_lat)        

    def extract_greedy_action(self, action):
        return action.drift > 0.5

    def select_features(self, features, features_vec):
        return features_vec

class RacingAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, features=TrackFeatures, **kwargs)
