# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import torch
from torch.distributions import Bernoulli, Normal, Categorical, OneHotCategorical
from .action_nets import LinearWithTanh, LinearWithSigmoid
from .rewards import steering_angle_reward, speed_reward

class BaseActor:

    def __init__(self, action_net, train=None, reward_type=None, sample_type=None):
        self.action_net = action_net.cpu().eval() if train != True else action_net
        self.train = train
        self.reward_type = reward_type
        self.sample_type = sample_type

    def copy(self, action_net):
        return self.__class__(action_net, train=self.train, reward_type=self.reward_type)

    def sample(self, *args):
        if self.sample_type == "bernoulli":
            return self.sample_bernoulli(*args)
        elif self.sample_type == "normal":
            return self.sample_normal(*args)
        elif self.sample_type == "categorical":
            return self.sample_categorical(*args)
        elif self.sample_type == "one_hot_categorical":
            return self.sample_one_hot_categorical(*args)
        raise Exception("Unknown sample type")

    def log_prob(self, *args, actions):
        input = args[0]
        if self.action_net.activation == "Tanh" or \
           self.action_net.activation == "Hardtanh":
            input = (input + 1) / 2
        if self.sample_type == "bernoulli":            
            dist = Bernoulli(probs=input) if self.action_net.activation != None else Bernoulli(logits=input)
            return dist.log_prob(actions)
        elif self.sample_type == "normal":
            dist = Normal(*args)
            return dist.log_prob(actions)
        elif self.sample_type == "categorical":
            dist = Categorical(probs=input) if self.action_net.activation != None else Categorical(logits=input)
            return dist.log_prob(actions)
        elif self.sample_type == "one_hot_categorical":
            dist = OneHotCategorical(probs=input) if self.action_net.activation != None else OneHotCategorical(logits=input)
            
            value = dist.log_prob(actions)
            print("one hot probs", actions, value)
            return value
        raise Exception("Unknown sample type")

    def sample_bernoulli(self, output):
        if self.action_net.activation == "Tanh" or \
           self.action_net.activation == "Hardtanh":
            output = (output + 1) / 2
        if self.action_net.activation != None:
            output = Bernoulli(probs=output).sample()
        else:
            output = Bernoulli(logits=output).sample()
        return output

    def sample_normal(self, location, scale):
        output = Normal(loc=location, scale=scale).sample()
        return output

    def sample_categorical(self, probs):
        if self.action_net.activation != None:
            output = OneHotCategorical(probs=probs).sample()
        else:
            output = Categorical(logits=probs).sample()
        return output

    def sample_one_hot_categorical(self, probs):
        if self.action_net.activation != None:
            output = OneHotCategorical(probs=probs).sample()
        else:
            output = OneHotCategorical(logits=probs).sample()
        return output

    def select_features(self, state_features):

        # this is only called for top level actors; nested actors are given features directly from their ancestors
        pass
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
        super().__init__(LinearWithSigmoid(3, 2, bias=True) if action_net is None else action_net, train=train, sample_type="bernoulli", **kwargs)        

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
