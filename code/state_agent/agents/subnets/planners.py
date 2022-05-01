# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/25/2022

from sre_parse import CATEGORIES
import torch
import numpy as np
from torch.nn import functional as F

from .features import PUCK_RADIUS
from .action_nets import LinearForNormalAndStd, CategoricalSelection
from .actors import BaseActor
from .rewards import MAX_DISTANCE, MAX_STEERING_ANGLE_REWARD, continuous_causal_reward, MAX_SOCCER_DISTANCE_REWARD, steering_angle_reward

class PlayerPuckGoalPlannerActor(BaseActor):

    LABEL_INDEX = 3
    FEATURES = 3
    CATEGORIES = 3
    HIDDEN_STATES = 5
    DELTA_STEERING_ANGLE_OUTPUT = 0
    DELTA_SPEED_OUTPUT = 1
    TARGET_SPEED_OUTPUT = 2

    acceleration = True

    def __init__(self, speed_net, steering_net, action_net=None, train=None, **kwargs):
        # Steering action_net
        # inputs: 
        #   player-puck distance
        #   puck-goal distance
        #   player-puck angle 
        #   puck-goal angle
        #   player velocity
        #  
        # categorical labels: 3
        #
        # outputs:
        #   delta steering angle
        #   delta speed
        #   target speed 
        
        super().__init__(CategoricalSelection(
            self.LABEL_INDEX, 
            self.FEATURES, 
            n_inputs=self.LABEL_INDEX,
            n_outputs=self.CATEGORIES,
            n_hidden=self.HIDDEN_STATES, 
            bias=True
        ) if action_net is None else action_net, train=train, sample_type="categorical")
        self.speed_net = speed_net
        self.steering_net = steering_net

    def copy(self, action_net):
        return self.__class__(self.speed_net, self.steering_net, action_net, train=self.train)

    def __call__(self, action, f, train=False, **kwargs):
        
        output = self.action_net(f)

        # print(output)

        if self.train is not None:
            train = self.train

        #assert(self.action_net.training == train)            
        if train:       
            # choose a set of labels by sampling
            sample = self.sample(output)
            output = self.action_net.choose(sample, f)
    
        # the subnetworks are not trained
        self.invoke_subnets(action, output, **kwargs)

        #print(action.acceleration, action.brake)
        
        return output

    def invoke_subnets(self, action, input, **kwargs):
        # steering direction - raw output        
        self.steering_net(action, input[[self.DELTA_STEERING_ANGLE_OUTPUT]], **kwargs)
        # delta speed, target speed - raw output
        self.speed_net(action, input, **kwargs)           
        

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
    
    def extract_greedy_action(self, action, f):        
        output = self.action_net(f)
        # the greedy action here is to only train the categorical index        
        return self.action_net.get_index(output)
        
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

        return torch.Tensor([
            pp_dist,
            #goal_dist,
            pp_angle,
            #ppg_angle,
            speed,

            # 1st label - behind the cart
            0.0, # steering directly behind the cart
            -10.0, # negative delta, ie reverse as fast as possible 
            -10.0, # negative target speed, ie reverse

            # 2nd label - go towards the puck 
            pp_angle,
            delta_speed,
            target_speed,

            # 3rd label - steer the puck towards the goal
            counter_steer_angle,
            delta_speed,
            target_speed,
            
        ])
    
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
