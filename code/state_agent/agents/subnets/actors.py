# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import torch
from .base_actors import BaseActor, LinearWithTanh, LinearWithSigmoid
from .rewards import steering_angle_reward, speed_reward

class SteeringActor(BaseActor):
    
    def __init__(self, action_net=None, train=None, **kwargs):
        # Steering action_net
        # inputs: delta steering angle
        super().__init__(LinearWithTanh(1, 1) if action_net is None else action_net, train=train, sample_type="bernoulli")

    def __call__(self, action, f, train=False, **kwargs):
        output = self.action_net(f)
        if self.train is not None:
            train = self.train
        if train:                        
            action.steer = self.sample(output) * 2 - 1
        else:
            action.steer = output[0] # raw output
        return action

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):
        current_angle = selected_features_curr
        next_angle = selected_features_next
        return steering_angle_reward(current_angle, next_angle)        
    
    def extract_greedy_action(self, action, *args, **kwargs):
        return [action.steer > 0]

    def select_features(self, features, features_vec):
        delta_steering_angle = features.select_player_puck_angle(features_vec)
        return torch.Tensor([
            delta_steering_angle
        ])

class DriftActor(BaseActor):
    
    def __init__(self, action_net=None, train=None, **kwargs):
        # Steering action_net
        # inputs: delta steering angle, delta lateral distance        
        super().__init__(LinearWithSigmoid(2, 1) if action_net is None else action_net, train=train, sample_type="bernoulli")

    def __call__(self, action, f, train=True, **kwargs):        
        output = self.action_net(f)
        if self.train is not None:
            train = self.train
        if train:            
            action.drift = self.sample(output) > 0.5
        else:
            # drift is a binary value
            action.drift = output[0] > 0.5
        
        return action

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):
        [current_angle] = selected_features_curr
        [next_angle] = selected_features_next        
        return steering_angle_reward(current_angle, next_angle)
        
    def extract_greedy_action(self, action, *args, **kwargs):
        return [action.drift > 0.5]

    def select_features(self, features, features_vec):
        delta_steering_angle = features.select_player_puck_angle(features_vec)        
        return torch.tensor([
            delta_steering_angle
        ])

class SpeedActor(BaseActor):

    acceleration = True

    def __init__(self, action_net=None, train=None, **kwargs):
        # inputs:         
        #   delta steering angle
        #   delta speed
        #   target speed
        # outputs:
        #   acceleration (0-1)
        #   brake (boolean)
        super().__init__(LinearWithSigmoid(3, 2) if action_net is None else action_net, train=train, sample_type="bernoulli", **kwargs)        

    def __call__(self, action, f, train=True, **kwargs):  
        output = self.action_net(f) 
        if self.train is not None:
            train = self.train
        if train:
            sample = self.sample(output)
            action.acceleration = sample[0]
            action.brake = sample[1] > 0.5
        else:            
            action.acceleration = output[0]
            # brake is a binary value
            action.brake = output[1] > 0.5
        
        return action

    def select_features(self, features, features_vec):
        delta_steering_angle = features.select_player_puck_angle(features_vec)        
        delta_speed = features.select_delta_speed(features_vec)        
        target_speed = features.select_target_speed(features_vec)        
        return torch.tensor([
            delta_steering_angle,
            delta_speed,
            target_speed
        ])

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):

        [_, current_speed, _] = selected_features_curr
        [_, next_speed, next_target_speed] = selected_features_next

        reward = [speed_reward(current_speed, next_speed)] * 2

        accel = round(action.acceleration, 5)
        
        if next_speed < -0.5:
            if accel > 0:
                reward[0] = -1 # should not accelerate if velocity should decrease!

            # Note: The car can slow down without having to brake... so we don't change the reward if the braking isn't enabled
        
        if next_speed > 0.5:

            if accel == 0:
                reward[0] = -1 # should be accelerating!
            
            if action.brake == True:
                reward[1] = -1 # the break should not be enabled

        if next_target_speed < 0.0:
            # the car should be going in reverse
            if accel != 0:
                reward[0] = -1 # acceleration should be 0

            if action.brake == False:
                reward[1] = -1 # brake should be True to move backwards

        elif next_speed == 0:

            if action.brake == True:
                reward[1] = -1 # brake should be False so the car will not move!

        return reward

    def extract_greedy_action(self, action, *args, **kwargs):
        return [
            # train acceleration to speed up or slow down (1, 0)
            action.acceleration > 0.5,
            action.brake > 0.5
        ]
