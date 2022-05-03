# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/25/2022

import torch
import numpy as np
from torch.nn import functional as F

from .features import PUCK_RADIUS, SoccerFeatures
from .action_nets import BooleanClassifier, Selection
from .actors import BaseActor
from .rewards import MAX_DISTANCE, MAX_STEERING_ANGLE_REWARD, continuous_causal_reward, MAX_SOCCER_DISTANCE_REWARD, steering_angle_reward

class Classifier(BaseActor):

    def __init__(self, features, range, action_net=None, train=None, **kwargs):
        super().__init__(BooleanClassifier(
                n_inputs=len(features),          
                n_hidden=1, 
                bias=False,
                range=range
            ) if action_net is None else action_net, train=train, sample_type="bernoulli")        
        self.feature_indicies = features
        
    def select_features(self, features, features_vec):
        return features.select_indicies(self.feature_indicies, features_vec)
    
    def extract_greedy_action(self, action, f):   
        output =self.action_net(f)
        return (output > 0.5).float().item()

class HeadToPuckClassifier(Classifier):

    FEATURES = [SoccerFeatures.PLAYER_PUCK_DISTANCE]

    def __init__(self, range, **kwargs):
        # inputs: 
        #   player-puck distance
        #  
        # outputs:
        #   confidence score (0-1)
        super().__init__(self.FEATURES, range, **kwargs)
             
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):

        selected_features_next = selected_features_next[self.action_net.range[0]:self.action_net.range[1]]
        selected_features_curr = selected_features_curr[self.action_net.range[0]:self.action_net.range[1]]

        (c_pp_dist) = selected_features_curr
        if greedy_action == 1:
            return 1 if c_pp_dist >= 0 else -1            
        else:
            return 1 if c_pp_dist < 0 else -1  

class PuckToGoalClassifier(Classifier):

    FEATURES = [SoccerFeatures.PLAYER_PUCK_DISTANCE]

    def __init__(self, range, action_net=None, train=None, **kwargs):
        # inputs: 
        #   player-puck distance
        #  
        # outputs:
        #   confidence score (0-1) 
        super().__init__(self.FEATURES, range, **kwargs)
            
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):

        selected_features_next = selected_features_next[self.action_net.range[0]:self.action_net.range[1]]
        selected_features_curr = selected_features_curr[self.action_net.range[0]:self.action_net.range[1]]

        (c_pp_dist) = selected_features_curr
        if greedy_action == 1:
            return 1 if c_pp_dist < 0 else -1            
        else:
            return 1 if c_pp_dist >= 0 else -1  

class PlayerPuckGoalPlannerActor(BaseActor):

    LABEL_INDEX = 3
    LABEL_FEATURES = 3
    DELTA_STEERING_ANGLE_OUTPUT = 0
    DELTA_SPEED_OUTPUT = 1
    TARGET_SPEED_OUTPUT = 2

    CLASSIFIER_HEAD_TO_PUCK = 0
    CLASSIFIER_PUCK_TO_GOAL = 1

    acceleration = True

    def __init__(self, speed_net, steering_net, action_net=None, train=None, **kwargs):
        # Steering action_net
        # inputs: 
        #   player-puck distance
        #   player-puck angle 
        #   player speed
        #  
        # categorical labels: 3
        #
        # outputs:
        #   delta steering angle
        #   delta speed
        #   target speed 
        
        classifiers = [
            HeadToPuckClassifier,
            PuckToGoalClassifier
        ]

        self.ranges = ranges = []
        index = 0
        for c in classifiers:
            ranges.append((index, len(c.FEATURES) + index))
            index += len(c.FEATURES)
        
        classifiers = [
            c(ranges[idx], action_net=action_net.classifiers[idx] if action_net else None) for idx, c in enumerate(classifiers)
        ]
        super().__init__(Selection(
            list(map(lambda x: x.action_net, classifiers)),
            self.ranges[-1][1],
            self.LABEL_FEATURES,
        ) if action_net is None else action_net, train=train, sample_type="bernoulli")
        self.classifiers = classifiers
        self.speed_net = speed_net
        self.steering_net = steering_net
        
    def copy(self, action_net):
        return self.__class__(self.speed_net, self.steering_net, action_net, train=self.train)

    def __call__(self, action, f, train=False, **kwargs):
        
        call_output = outputs = self.action_net(f)
        if train:            
            # sample between the cases using softmax
            call_output = torch.Tensor([c.sample(call_output[idx]) for idx, c in enumerate(self.classifiers)])
            # get labels
            y = self.action_net.get_labels(f)
            outputs = self.action_net.choose(call_output, y)
        
        # the subnetworks are not trained
        self.invoke_subnets(action, outputs, **kwargs)

        #print(action.acceleration, action.brake)        
        return call_output

    def invoke_subnets(self, action, input, **kwargs):
        # steering direction - raw output        
        self.steering_net(action, input[[self.DELTA_STEERING_ANGLE_OUTPUT]], **kwargs)
        # delta speed, target speed - raw output
        self.speed_net(action, input, **kwargs)           
        

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):
        rewards = [c.reward(action, greedy_action[idx], selected_features_curr, selected_features_next) for idx, c in enumerate(self.classifiers)]
        return rewards

    """

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):
        (c_pp_dist, c_pp_angle, c_speed, *_) = selected_features_curr
        (n_pp_dist, n_pp_angle, n_speed, *_) = selected_features_next

        reward = 0

        # is the puck outside of the maximum steering angle?
        #puck_outside_angle_fast = torch.abs(c_pp_angle) > 0.35

        # are we likely stuck next to a wall?
        puck_outside_angle_slow = torch.abs(c_pp_angle) > 0.35 and c_speed < 0.02

        #if puck_outside_angle_fast:
        if puck_outside_angle_slow:

            reward = torch.abs(c_pp_angle) if greedy_action == 0 else -torch.abs(c_pp_angle)

        # is the player next to the puck?
        elif c_pp_dist >= 0:
            reward = c_pp_dist if greedy_action == 1 else -c_pp_dist
            reward = reward / MAX_DISTANCE
            
        elif c_pp_dist < 0:
            reward = -c_pp_dist if greedy_action == 2 else c_pp_dist
            reward = reward / PUCK_RADIUS           

        #print("planner reward ", greedy_action, reward, c_pp_dist)

        # return rewards for each case
        return reward
    """
    
    def extract_greedy_action(self, action, f):                
        return [c.extract_greedy_action(action, f) for c in self.classifiers]
                
    def select_features(self, features, features_vec):
        pp_dist = features.select_player_puck_distance(features_vec)
        goal_dist = features.select_puck_goal_distance(features_vec)
        pp_angle = features.select_player_puck_angle(features_vec)
        ppg_angle = features.select_player_puck_angle(features_vec)
        counter_steer_angle = features.select_player_puck_countersteer_angle(features_vec)
        behind_angle = features.select_behind_player_angle(features_vec)
        speed = features.select_speed(features_vec)
        delta_speed = features.select_delta_speed(features_vec)
        delta_speed_behind = features.select_delta_speed_behind(features_vec)
        target_speed = features.select_target_speed(features_vec)
        target_speed_behind = features.select_speed_behind(features_vec)
                
        #print("Speed", speed)

        labels = torch.Tensor([
            # 1st label - behind the cart
            #0.0, # steering directly behind the cart
            #-10.0, # negative delta, ie reverse as fast as possible 
            #-target_speed, # negative target speed, ie reverse

            # 2nd label - go towards the puck 
            pp_angle,
            delta_speed,
            target_speed,

            # 3rd label - steer the puck towards the goal
            counter_steer_angle,
            delta_speed,
            target_speed,
            
        ])

        features = [classifier.select_features(features, features_vec) for classifier in self.classifiers]
        features.append(labels)

        return torch.concat(features)

    def log_prob(self, *args, actions):
        input = args[0]
        return torch.concat([c.log_prob(input[:,idx], actions=actions[:,idx]).unsqueeze(1) for idx, c in enumerate(self.classifiers)], dim=1)
    
"""
The goal of the fine tuned planner is to use the outputs of the base planner categories as the 'mean' 
of a stochastic monte-carlo search for finding the best target angle and speed to optimize an objective function.

In simpler terms, it will train by generating noise to offset the base planners output and learn what offsets to apply before passing outputs to subnetworks.
"""
class PlayerPuckGoalFineTunedPlannerActor(PlayerPuckGoalPlannerActor):

    def __init__(self, speed_net, steering_net, action_net=None, train=None, **kwargs):
        # Steering action_net
        # inputs: 
        #   player-puck distance
        #   puck-goal distance
        #   player-puck angle 
        #   puck-goal angle
        #   player velocity
        # outputs:
        #   delta steering angle (mean)
        #   delta speed (mean)
        #   target speed (mean)
        #   delta steering angle (std)
        #   delta speed (std)
        #   target speed (std)


        super().__init__(speed_net, steering_net, action_net, train=train, sample_type="normal")
        
    def __call__(self, action, f, train=False, **kwargs):        
        output = self.action_net(f)

        if self.train is not None:
            train = self.train
        if train:                      
            output[0:3] = self.sample(output[0:3], output[3:6])            
            
        # the subnetworks are not trained
        
        # steering direction - raw output        
        self.steering_net(action, output[[0]], **kwargs)
        # delta speed, target speed - raw output
        self.speed_net(action, output[0:3], **kwargs)           
        
        return output

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):
        (c_pp_dist, c_goal_dist, *_) = selected_features_curr
        (n_pp_dist, n_goal_dist, *_) = selected_features_next
        reward_puck_dist = continuous_causal_reward(c_pp_dist / MAX_DISTANCE * MAX_SOCCER_DISTANCE_REWARD, n_pp_dist / MAX_DISTANCE * MAX_SOCCER_DISTANCE_REWARD, 1.0, MAX_SOCCER_DISTANCE_REWARD)
        reward_goal_dist = continuous_causal_reward(c_goal_dist / MAX_DISTANCE * MAX_SOCCER_DISTANCE_REWARD, n_goal_dist / MAX_DISTANCE * MAX_SOCCER_DISTANCE_REWARD, 0.1, MAX_SOCCER_DISTANCE_REWARD)
        return [(reward_goal_dist + reward_puck_dist) / 2] * 3
    
    def extract_greedy_action(self, action, f):        
        # determine the steering direction and speed        
        output = self.action_net(f)        
        return output[0:3].detach().numpy()   

    def select_features(self, features, features_vec):
        pp_dist = features.select_player_puck_distance(features_vec)
        goal_dist = features.select_puck_goal_distance(features_vec)
        pp_angle = features.select_player_puck_angle(features_vec)
        ppg_angle = features.select_player_puck_goal_angle(features_vec)
        speed = features.select_speed(features_vec)
        
        return torch.Tensor([
            pp_dist,
            goal_dist,
            pp_angle,
            ppg_angle,
            speed
        ])

    def log_prob(self, output, actions):
        retval = super().log_prob(output[:, 0:3], torch.abs(output[:, 3:6]), actions=actions)
        return retval
