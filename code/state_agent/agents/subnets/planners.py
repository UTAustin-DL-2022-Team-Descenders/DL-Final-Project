# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/25/2022

import torch
import numpy as np
import copy
import os
from torch.nn import functional as F

from state_agent.agents.subnets.agents import Action

from state_agent.agents.subnets.features import MIN_WALL_SPEED, NEAR_WALL_OFFSET, NEAR_WALL_STD, PUCK_RADIUS, MAX_SPEED, SoccerFeatures
from state_agent.agents.subnets.action_nets import BooleanClassifier, LinearNetwork, Selection, LinearWithTanh
from state_agent.agents.subnets.actors import BaseActor
from state_agent.agents.subnets.rewards import MAX_DISTANCE, MAX_STEERING_ANGLE_REWARD, continuous_causal_reward, MAX_SOCCER_DISTANCE_REWARD, continuous_causal_reward_ext, steering_angle_reward

class Classifier(BaseActor):

    def __init__(self, features, range, action_net=None, train=None, **kwargs):
        super().__init__(BooleanClassifier(
                n_inputs=len(features),                                          
                range=range,
                **kwargs
            ) if action_net is None else action_net, train=train, sample_type="bernoulli")        
        self.feature_indicies = features
        
    def select_features(self, features):
        return features.select_indicies(self.feature_indicies)
    
    def get_selected_features(self, selected_features_curr, selected_features_next):
        selected_features_next = selected_features_next[self.action_net.range[0]:self.action_net.range[1]] if selected_features_next else None
        selected_features_curr = selected_features_curr[self.action_net.range[0]:self.action_net.range[1]]
        return selected_features_curr, selected_features_next

    def extract_greedy_action(self, action, f):   
        output =self.action_net(f)
        # classifier uses Tanh, so use output > 0
        return (output > 0).float().item()

class HeadToPuckClassifier(Classifier):

    FEATURES = [SoccerFeatures.PLAYER_PUCK_DISTANCE]

    def __init__(self, range, **kwargs):
        # inputs: 
        #   player-puck distance
        #  
        # outputs:
        #   confidence score (0-1)
        super().__init__(self.FEATURES, range, n_hidden=1, bias=False, **kwargs)
             
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):

        selected_features_curr, selected_features_next = self.get_selected_features(selected_features_curr, selected_features_next)

        (c_pp_dist) = selected_features_curr
        # return the boolean value as the 'label' for classification
        return c_pp_dist >= 0

        
class PuckToGoalClassifier(Classifier):

    FEATURES = [SoccerFeatures.PLAYER_PUCK_DISTANCE]

    def __init__(self, range, action_net=None, train=None, **kwargs):
        # inputs: 
        #   player-puck distance
        #  
        # outputs:
        #   confidence score (0-1) 
        super().__init__(self.FEATURES, range, n_hidden=1, bias=False, **kwargs)
            
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):

        selected_features_curr, selected_features_next = self.get_selected_features(selected_features_curr, selected_features_next)
        
        #print("PuckToGoal", selected_features_curr, greedy_action)

        (c_pp_dist) = selected_features_curr
        # return the boolean value as the 'label' for classification
        return c_pp_dist < 0

class RecoverTowardsPuck(Classifier):

    FEATURES = [        
        SoccerFeatures.SPEED,        
        SoccerFeatures.PREVIOUS_SPEED,
        SoccerFeatures.PLAYER_PUCK_ANGLE
    ]

    def __init__(self, range, action_net=None, train=None, **kwargs):
        # inputs: 
        #   player-puck distance
        #  
        # outputs:
        #   confidence score (0-1) 
        super().__init__(self.FEATURES, range, n_hidden=5, bias=True, **kwargs)
            
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):

        selected_features_curr, selected_features_next = self.get_selected_features(selected_features_curr, selected_features_next)

        (c_speed, c_prev_speed, c_pp_angle) = selected_features_curr

        # are we likely stuck next to a wall?
        puck_stuck = c_speed < MIN_WALL_SPEED and c_prev_speed < MIN_WALL_SPEED and np.abs(c_pp_angle) > 0.35

        # return the boolean value as the 'label' for classification
        return puck_stuck
        

class PlayerPuckGoalPlannerActor(BaseActor):

    LABEL_FEATURES = 3
    DELTA_STEERING_ANGLE_OUTPUT = 0
    DELTA_SPEED_OUTPUT = 1
    TARGET_SPEED_OUTPUT = 2

    CLASSIFIER_HEAD_TO_PUCK = 0
    CLASSIFIER_PUCK_TO_GOAL = 1
    CLASSIFIER_STUCK_AGAINST_WALL = 2

    OUT_FEATURE_DELTA_ANGLE = 0,
    OUT_FEATURE_DELTA_SPEED = 1,
    OUT_FEATURE_TARGET_SPEED = 2,

    def __init__(self, action_net=None, train=None, **kwargs):
        
        classifiers = [
            HeadToPuckClassifier,
            PuckToGoalClassifier,
            RecoverTowardsPuck
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
            self.LABEL_FEATURES
        ) if action_net is None else action_net, train=train, sample_type="bernoulli")
        self.selection_bias = torch.Tensor([0.0, 0.0, 0.05]) # boost the 'reccovery' case otherwise it will generally be overshadowed because it is a rare event
        self.classifiers = classifiers

        self.model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "modules", "planner")

        # Set model name for saving and loading action net
        self.model_name = "planner_net"
        
    #def set_feature_offset(self, offset):
        # offset is the sum of:
        # the final offset of all the classifier features (determined by the last range value)
        # the sum of all label features (each classifier has LABEL_FEATURES labels)
    #    self.action_net.range = (offset, offset + self.ranges[-1][1] + self.LABEL_FEATURES * len(self.ranges))

    def copy(self, action_net):
        return self.__class__(action_net, train=self.train)

    def __call__(self, action, f, extractor:SoccerFeatures=None, train=False, **kwargs):
        
        call_output = outputs = self.action_net(f, self.selection_bias if train == False else None)
        if train:                        
            call_output = torch.Tensor([c.sample(call_output[idx]) for idx, c in enumerate(self.classifiers)])
            # get labels
            y = self.action_net.get_labels(f)
            outputs = self.action_net.choose(call_output, y, None) # bias is None for training
        
        # set the output features
        extractor.set_features([
            SoccerFeatures.STEERING_ANGLE,
            SoccerFeatures.DELTA_SPEED,
            SoccerFeatures.TARGET_SPEED
        ], outputs)

        extractor.set_features([
            SoccerFeatures.PLANNER_CHOICE,
        ], [self.action_net.last_choice])

        #print(action.acceleration, action.brake)        
        return call_output

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):
        #print("reward: ", greedy_action)
        rewards = [1.0 for idx, c in enumerate(self.classifiers)]
        return rewards
    
    def extract_greedy_action(self, action, f):   

        # WTH is going on here? Instead of using the greedy action in the probability, use the rewards as a 'label'.
        # Bernoulli calls BCELossWithLogits using the actions as the 'labels', so using the rewards helps the
        # planner to classify these cases far more easily than the typical policy gradient approach.
        
        rewards = [c.reward(action, c.extract_greedy_action(action, f), f, None) for idx, c in enumerate(self.classifiers)]
        return rewards

        #return [c.extract_greedy_action(action, f) for c in self.classifiers]
                
    def select_features(self, features):
        pp_angle = features.select_player_puck_angle()
        counter_steer_angle = features.select_player_puck_countersteer_angle()
        speed = features.select_speed()
        delta_speed = features.select_delta_speed()
        target_speed = features.select_target_speed()
                
        #print("Speed", speed)

        labels = torch.Tensor([
            
            #  go towards the puck 
            pp_angle,
            delta_speed,
            target_speed,

            # steer the puck towards the goal
            counter_steer_angle,
            delta_speed,
            target_speed,
            
            # drive backwards
            -0.5 * np.sign(pp_angle), # steer in direction opposite of puck direction
            -target_speed - speed, # negative delta, ie reverse as fast as possible 
            -target_speed, # negative target speed, ie reverse

        ])

        features = [classifier.select_features(features) for classifier in self.classifiers]
        features.append(labels)

        features = torch.concat(features)

        return features

    def log_prob(self, *args, actions):
        input = args[0]
        return torch.concat([c.log_prob(input[:,idx], actions=actions[:,idx]).unsqueeze(1) for idx, c in enumerate(self.classifiers)], dim=1)

"""
The goal of the fine tuned planner is to use the outputs of the base planner categories as the 'mean'
of a stochastic monte-carlo search for finding the best target angle and speed to optimize an objective function.

In simpler terms, it will train by generating noise to offset the base planners output and learn what offsets to apply before passing outputs to subnetworks.
"""
class PlayerPuckGoalFineTunedPlannerActor(BaseActor):

    FEATURES = [    
        SoccerFeatures.PLAYER_PUCK_DISTANCE,            
        SoccerFeatures.SPEED,        
        SoccerFeatures.PREVIOUS_SPEED,
        SoccerFeatures.PLAYER_PUCK_ANGLE,
        SoccerFeatures.PUCK_GOAL_DISTANCE,
        SoccerFeatures.PUCK_GOAL_ANGLE,
        SoccerFeatures.PLANNER_CHOICE
    ]

    IN_FEATURE_SPEED = 1
    IN_FEATURE_PLANNER_CHOICE = 6
    IN_FEATURE_TARGET_SPEED = 7

    PLANNER_FEATURE_INDEX = len(FEATURES)
    FEATURE_INDEX = 5
    INPUTS = len(FEATURES)
    OUTPUTS = 2
    HIDDEN = 20
    
    def __init__(self, action_net=None, train=None, mode=None, **kwargs):
        # Steering action_net
        # inputs:
        #   choice index (provided by planner.last_choice) 
        #   puck-goal distance
        #   puck-goal angle
        #   [all of the inputs of the planner module]        
        # outputs:
        #   delta steering angle
        #   delta speed
        #   target speed
        
        #   delta steering angle (mean)
        #   delta speed (mean)
        #   target speed (mean)
        #   delta steering angle (std)
        #   delta speed (std)
        #   target speed (std)
        #super().__init__(action_net if action_net else LinearForNormalAndStd(n_inputs=self.FEATURES, n_outputs=self.OUTPUTS, index_start=self.FEATURES, bias=True), 
        #    train=train, sample_type="normal", **kwargs
        #) 
        self.stds = torch.Tensor([
            0.0001,
            0.0001,
            0.0001
        ])
        super().__init__(
            action_net if action_net else LinearNetwork(
                None,
                n_inputs=self.INPUTS, 
                n_outputs=self.OUTPUTS, 
                n_hidden=self.HIDDEN,
                bias=True
            ), 
            train=train, sample_type="bernoulli", **kwargs
        )     
        self.mode = mode  
        
    def __call__(self, action, f, train=False, **kwargs):  
        
        if self.train is not None:
            train = self.train

        offset = self.action_net(f)
                
        if train is not None:
            offset = self.sample(offset)

        if self.mode == "steering":
            # only train the steering
            output[1] = output[2] = 0
        elif self.mode == "speed":
            # only train the speed
            offset[0] = 0

        # the network will output an offset of the current speed, target speed needs to be determined from this
        offset = torch.tensor([
            offset[0],            
            offset[1],
            0
        ], device=offset.device)

        # current speed
        speed = self.get_input_feature(f, self.IN_FEATURE_SPEED)

        # add offset from the fine-tuned net
        output += offset
        # adjust the target speed accordingly
        output[PlayerPuckGoalPlannerActor.OUT_FEATURE_TARGET_SPEED] = offset[1] + speed

        self.planner_net.invoke_subnets(action, output, **kwargs)
        
        return offset
    
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next):
        (c_pp_dist, c_speed, c_prev_speed, c_pp_angle, c_goal_dist, c_pg_angle, *_) = selected_features_curr
        (n_pp_dist, n_speed, n_prev_speed, n_pp_angle, n_goal_dist, n_pg_angle, *_) = selected_features_next

        # get the planner choice
        choice = greedy_action[0]

        reward = 0

        #print("choice", choice)

        # reward the outcomes that minimizes the puck's distance to the goal
        if choice == PlayerPuckGoalPlannerActor.CLASSIFIER_PUCK_TO_GOAL:               
            reward = continuous_causal_reward_ext(c_goal_dist, n_goal_dist, 0.1, MAX_DISTANCE) / MAX_DISTANCE
        
        # reward the outcomes that minimizes both the player/puck to goal angle and distance to the puck       
        elif choice == PlayerPuckGoalPlannerActor.CLASSIFIER_HEAD_TO_PUCK:
            reward_dist = continuous_causal_reward_ext(c_pp_dist, n_pp_dist, 0.1, MAX_DISTANCE) / MAX_DISTANCE
            reward_angle = continuous_causal_reward_ext(torch.abs(c_pp_angle - c_pg_angle), torch.abs(n_pp_angle - n_pg_angle), 0.1, MAX_STEERING_ANGLE_REWARD) / MAX_STEERING_ANGLE_REWARD

            # weight the puck to goal higher
            reward = 0
            if self.mode == None or self.mode == "steering":
                reward += reward_dist * 0.25
            if self.mode == None or self.mode == "speed":
                reward += reward_angle
            reward /= 2
            
        elif choice == PlayerPuckGoalPlannerActor.CLASSIFIER_STUCK_AGAINST_WALL:
            reward = continuous_causal_reward_ext(c_speed, n_speed, 0.1, MAX_SPEED) / MAX_SPEED if n_speed < 0 else -1

        return [reward] * self.OUTPUTS
    
    def extract_greedy_action(self, action, f):        
        # determine the steering direction and speed  
        input = self.get_planner_features(f)
        action_new = Action()
        output = self.planner_net(action_new, input)
        output = self.action_net(self.get_input_features(f, output)).detach().numpy()
        return np.concatenate([
            np.array([self.planner_net.action_net.last_choice]),            
            output > 0, 
        ])

    def select_features(self, features, features_vec):
        return torch.concat([
            features.select_indicies(self.FEATURES, features_vec),
            self.planner_net.select_features(features, features_vec)
        ])
        
    def log_prob(self, output, actions):
        # skip the first value of the actions (it contains the planner choice)
        actions = actions[:,1:]
        retval = super().log_prob(output, actions=actions)
        return retval
