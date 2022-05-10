# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/25/2022

from typing import List
import torch
import numpy as np
import os
from torch.nn import functional as F
from state_agent.actors import BaseActorNetwork, Action
from state_agent.features import MIN_WALL_SPEED, NEAR_WALL_OFFSET, NEAR_WALL_STD, PUCK_RADIUS, MAX_SPEED, SoccerFeatures
from state_agent.action_nets import BooleanClassifier, LinearNetwork, Selection, LinearWithTanh
from state_agent.actors import BaseActor
from state_agent.rewards import MAX_DISTANCE, MAX_STEERING_ANGLE_REWARD, continuous_causal_reward, MAX_SOCCER_DISTANCE_REWARD, continuous_causal_reward_ext, steering_angle_reward

class ClassifierNetwork(torch.nn.Module):

    def __init__(self, features, range, n_hidden, bias):
        super().__init__()
        self.action_net = BooleanClassifier(
                n_inputs=len(features),                                          
                n_hidden=n_hidden,
                range=range,
                scale=None,
                bias=bias
        )
        self.feature_indicies = features

    def forward(self, x):
        out = self.action_net(x)
        return out

    def select_features(self, features: SoccerFeatures):
        return features.select_indicies(self.feature_indicies)
    
class Classifier(BaseActor):

    def __init__(self, features, range, n_hidden=None, bias=None, classifier_net=None, train=None):
        super().__init__(ClassifierNetwork(features, range, n_hidden=n_hidden, bias=bias) if classifier_net is None else classifier_net, train=train, sample_type="bernoulli")
        self.feature_indicies = features

    def get_actor_net(self) -> ClassifierNetwork:
        return self.actor_net  # type: ignore

    def get_selected_features(self, selected_features_curr, selected_features_next):
        range = self.get_actor_net().action_net.get_range()
        selected_features_next = selected_features_next[range[0]:range[1]] if selected_features_next else None
        selected_features_curr = selected_features_curr[range[0]:range[1]]
        return selected_features_curr, selected_features_next

    def extract_greedy_action(self, action, f):   
        output =self.actor_net(f)
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
             
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next, time):

        selected_features_curr, selected_features_next = self.get_selected_features(selected_features_curr, selected_features_next)

        (c_pp_dist) = selected_features_curr
        # return the boolean value as the 'label' for classification
        return c_pp_dist >= 0

        
class PuckToGoalClassifier(Classifier):

    FEATURES = [SoccerFeatures.PLAYER_PUCK_DISTANCE]

    def __init__(self, range, **kwargs):
        # inputs: 
        #   player-puck distance
        #  
        # outputs:
        #   confidence score (0-1) 
        super().__init__(self.FEATURES, range, n_hidden=1, bias=False, **kwargs)
            
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next, time):

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
            
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next, time):

        selected_features_curr, selected_features_next = self.get_selected_features(selected_features_curr, selected_features_next)

        (c_speed, c_prev_speed, c_pp_angle) = selected_features_curr

        # are we likely stuck next to a wall?
        puck_stuck = c_speed < MIN_WALL_SPEED and c_prev_speed < MIN_WALL_SPEED and np.abs(c_pp_angle) > 0.35

        # return the boolean value as the 'label' for classification
        return puck_stuck
        
class PlayerPuckGoalPlannerActorNetwork(BaseActorNetwork):

    def __init__(self, classifiers: List[ClassifierNetwork], label_index: int, number_features: int) -> None:
        action_net = Selection(
            label_index,
            number_features
        )
        super().__init__(action_net)
        self.classifiers = torch.nn.ModuleList(classifiers)

    @property
    def selection_action_net(self) -> Selection:
        return self.action_net # type: ignore

    def forward(self, action: Action, f: torch.Tensor, extractor: SoccerFeatures):
        outputs = self.call(action, f, extractor.selection_planner_bias())
        self.set_outgoing_features(outputs, extractor)
        return outputs

    def set_outgoing_features(self, outputs: torch.Tensor, extractor: SoccerFeatures):

        # set the input features for the downsteam networks
        extractor.set_features([
            extractor.STEERING_ANGLE,
            extractor.DELTA_SPEED,
            extractor.TARGET_SPEED
        ], outputs)

        extractor.set_features([
            extractor.PLANNER_CHOICE,
        ], self.selection_action_net.last_choice[None])

    def call(self, action: Action, f: torch.Tensor, bias: torch.Tensor):
        # run the classifiers to generate a set of scores as an index tensor
        index = self.get_index(f)
        # run the selection network to choose from the index and produce the choosen labels
        return self.action_net(f, index, bias)

    def get_index(self, input: torch.Tensor):
        index: List[torch.Tensor] = []

        for idx, classifier in enumerate(self.classifiers):
            index.append(classifier(input))
        return torch.cat(index, dim=1 if input.dim() > 1 else 0)

    def get_labels(self, x: torch.Tensor):
        return self.selection_action_net.get_labels(x)

    def choose(self, x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor):
        return self.selection_action_net.choose(x, y, bias)

    def select_features(self, features: SoccerFeatures):
        pp_attack_angle = features.select_player_puck_attack_angle()
        pp_angle = features.select_player_puck_angle()
        counter_steer_angle = features.select_player_puck_countersteer_angle()
        speed = features.select_speed()
        delta_speed = features.select_delta_speed()
        target_speed = features.select_target_speed()

        #print("Speed", speed)

        # adjust the puck target angle to head towards the angle of incidence

        # XXX might need to be fixed to address 'one-network' rule

        labels = torch.cat([

            #  go towards the puck
            pp_attack_angle[None],
            delta_speed[None],
            target_speed[None],

            # steer the puck towards the goal
            counter_steer_angle[None],
            delta_speed[None],
            target_speed[None],

            # drive backwards
            (-0.5 * torch.sign(pp_angle))[None], # steer in direction opposite of puck direction
            (-target_speed - speed)[None], # negative delta, ie reverse as fast as possible
            (-target_speed)[None], # negative target speed, ie reverse

        ])

        selected_features: List[torch.Tensor] = []

        for idx, classifier in enumerate(self.classifiers):
            cls: ClassifierNetwork = classifier # type: ignore
            selected_features.append(cls.select_features(features))
        selected_features.append(labels)

        out = torch.cat(selected_features)

        return out

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

    def __init__(self, actor_net: PlayerPuckGoalPlannerActorNetwork=None, train=None):
        
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
               c(ranges[idx], classifier_net=actor_net.classifiers[idx] if actor_net else None) for idx, c in enumerate(classifiers)
        ]

        if actor_net is None:
            actor_net = PlayerPuckGoalPlannerActorNetwork(
                list(map(lambda x: x.actor_net, classifiers)),
                self.ranges[-1][1],
                self.LABEL_FEATURES
            ) if actor_net is None else actor_net


        super().__init__(actor_net, train=train, sample_type="bernoulli")
        self.classifiers = classifiers

        self.model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                       "agents/subnets/modules", "planner")

        # Set model name for saving and loading action net
        self.model_name = "planner_net"
        
    @property
    def this_actor_net(self) -> PlayerPuckGoalPlannerActorNetwork:
        return self.actor_net # type: ignore

    def copy(self, action_net):
        return self.__class__(action_net, train=self.train)

    def __call__(self, action, f, extractor:SoccerFeatures, train=False, **kwargs):
        
        # call the network directly and apply outputs
        call_output = outputs = self.actor_net(action, f, extractor)

        if self.train is not None:
            train = self.train

        if train:
            call_output = torch.Tensor([c.sample(call_output[idx]) for idx, c in enumerate(self.classifiers)])
            # get labels
            y = self.this_actor_net.get_labels(f)

            # modify the outgoing features with the sampled choice instead
            outputs = self.this_actor_net.choose(call_output, y, None) # bias is None for training

            # apply modified outputs
            self.this_actor_net.set_outgoing_features(outputs, extractor)

        return call_output

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next, time):

        # the planner is not trained with policy gradients... it is trained with BCE. We assign a reward of 1 as the weight.
        rewards = [1.0 for idx, c in enumerate(self.classifiers)]
        return rewards
    
    def extract_greedy_action(self, action, f):   

        # The planner classifiers are not trained with policy gradients. They are trained with BCE. This is done here as
        # a way to keep in line with the policy gradient training framework

        # Instead of using the greedy action in the probability, use the rewards as a 'label'.
        # Bernoulli calls BCELossWithLogits using the actions as the 'labels', so using the rewards helps the
        # planner to classify these cases far more easily than the typical policy gradient approach.
        
        rewards = [c.reward(action, c.extract_greedy_action(action, f), f, None) for idx, c in enumerate(self.classifiers)]
        return rewards

        #return [c.extract_greedy_action(action, f) for c in self.classifiers]
                
    def log_prob(self, *args, actions):
        input = args[0]
        return torch.cat([c.log_prob(input[:,idx], actions=actions[:,idx]).unsqueeze(1) for idx, c in enumerate(self.classifiers)], dim=1)

class PlayerPuckGoalFineTunedPlannerActorNetwork(BaseActorNetwork):

    OUTPUTS = 2
    HIDDEN = 40

    MODE_SPEED = 0
    MODE_STEERING = 1
    MODE_BOTH = 2

    def __init__(self, mode: int):
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


        self.FEATURES = [
            SoccerFeatures.PLAYER_PUCK_DISTANCE,
            SoccerFeatures.SPEED,
            SoccerFeatures.PREVIOUS_SPEED,
            SoccerFeatures.TARGET_SPEED,
            SoccerFeatures.PLAYER_PUCK_ANGLE,
            SoccerFeatures.PUCK_GOAL_DISTANCE,
            SoccerFeatures.PUCK_GOAL_ANGLE,
            SoccerFeatures.PLAYER_PUCK_COUNTER_STEER_ANGLE,
            SoccerFeatures.PLANNER_CHOICE
        ]

        self.OUTPUT_STEERING_OFFSET = 0
        self.OUTPUT_SPEED_OFFSET = 1

        super().__init__(LinearNetwork(
            None,
            n_inputs=len(self.FEATURES),
            n_outputs=self.OUTPUTS,
            n_hidden=self.HIDDEN,
            bias=True,
            scale=None,
            range=None
        ))

        # Torchscript nonsense
        self.MODE_SPEED = 0
        self.MODE_STEERING = 1
        self.MODE_BOTH = 2

        self.mode = torch.tensor(mode)
        #self.register_buffer("mode", self.mode)

    def forward(self, action: Action, f: torch.Tensor, extractor: SoccerFeatures):
        outputs = self.call(action, f)
        return self.set_output_features(outputs, extractor)

    def call(self, action: Action, f: torch.Tensor):
        return self.action_net(f)

    def set_output_features(self, outputs: torch.Tensor, extractor: SoccerFeatures):

        # zero out features when the network is only trained for certain output behaviors
        if self.mode == self.MODE_STEERING:

            # only train the steering
            outputs[self.OUTPUT_SPEED_OFFSET] = 0

        elif self.mode == self.MODE_SPEED:
            # only train the speed
            outputs[self.OUTPUT_STEERING_OFFSET] = 0

        # target speed
        target_speed = extractor.select_target_speed()
        speed = extractor.select_speed()

        # add offset from the fine-tuned net
        # set the output features
        extractor.set_features([
            extractor.STEERING_ANGLE,
            extractor.TARGET_SPEED
        ], extractor.select_indicies([
            extractor.STEERING_ANGLE,
            extractor.TARGET_SPEED
        ]) + outputs)

        # adjust the delta speed accordingly because the action network only outputs target speeds not offsets
        if self.mode != self.MODE_STEERING:
            extractor.set_features([
                extractor.DELTA_SPEED
            ], torch.as_tensor([
                target_speed + outputs[1] - speed
            ]))

        return outputs


    def select_features(self, features):
        return features.select_indicies(self.FEATURES)

"""
The goal of the fine tuned planner is to use the outputs of the base planner categories as the 'mean'
of a stochastic monte-carlo search for finding the best target angle and speed to optimize an objective function.

In simpler terms, it will train by generating noise to offset the base planners output and learn what offsets to apply before passing outputs to subnetworks.
"""
class PlayerPuckGoalFineTunedPlannerActor(BaseActor):

    IN_FEATURE_SPEED = 1
    IN_FEATURE_PLANNER_CHOICE = 8
    IN_FEATURE_TARGET_SPEED = 3

    OUT_FEATURE_SPEED_OFFSET = 1

    def __init__(self, actor_net=None, train=None, mode:int=PlayerPuckGoalFineTunedPlannerActorNetwork.MODE_BOTH, **kwargs):
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
        
        self.stds = torch.Tensor([
            0.0001,
            0.0001,
            0.0001
        ])
        super().__init__(
            actor_net if actor_net else PlayerPuckGoalFineTunedPlannerActorNetwork(mode), train=train, sample_type="bernoulli", **kwargs
        )
        self.model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "agents/subnets/modules", "ft_planner")

        # Set model name for saving and loading action net
        self.model_name = "ft_planner_net"
        
    @property
    def this_actor_net(self) -> PlayerPuckGoalFineTunedPlannerActorNetwork:
        return self.actor_net # type: ignore

    def __call__(self, action, f, extractor:SoccerFeatures, train=False, **kwargs):
        
        offset = self.action_net(f, action, extractor)

        if self.train:
            train = self.train
        if train:
            offset = (self.sample(offset) * 2 - 1)
            offset[1] *= MAX_SPEED / 4

            self.this_actor_net.set_output_features(offset, extractor)

        # print(offset, "speed", speed, "speed (ext)", extractor.select_speed(), "target speed", extractor.select_target_speed(), "delta speed", extractor.select_delta_speed())
        
        return offset

    def copy(self, actor_net):
        return self.__class__(actor_net, train=self.train)
    
    def reward(self, action, greedy_action, selected_features_curr, selected_features_next, time):
        (c_pp_dist, c_speed, c_prev_speed, c_target_speed, c_pp_angle, c_goal_dist, c_pg_angle, c_counter, c_choice, *_) = selected_features_curr
        (n_pp_dist, n_speed, n_prev_speed, n_target_speed, n_pp_angle, n_goal_dist, n_pg_angle, n_counter, n_choice, *_) = selected_features_next

        reward = 0

        #print("choice", choice)

        # reward the outcomes that minimizes the puck's distance to the goal
        if c_choice == PlayerPuckGoalPlannerActor.CLASSIFIER_PUCK_TO_GOAL or \
           c_choice == PlayerPuckGoalPlannerActor.CLASSIFIER_HEAD_TO_PUCK:


            # reward the outcomes that minimizes both the player/puck to goal angle and distance to the puck
            reward_goal = continuous_causal_reward_ext(c_goal_dist, n_goal_dist, 0.1, 0.5, MAX_DISTANCE) / MAX_DISTANCE
            reward_dist = continuous_causal_reward_ext(c_pp_dist, n_pp_dist, 0.1, 0.5, MAX_DISTANCE) / MAX_DISTANCE

            # scale reward by time, asymmetrically (negative rewards get worse with more time, positive rewards get better with less time)

            reward_goal = reward_goal * (time if reward_goal < 0 else 1/time)
            reward_dist = reward_dist * (time if reward_dist < 0 else 1/time)

            # magnify mistakes that move the puck away from the goal
            #if reward_goal < 0:
            #    reward_goal = - np.power(reward_goal, 2)
            #if reward_dist < 0:
            #    reward_dist = - np.power(reward_dist, 2)

            reward = 0
            reward += (reward_goal + reward_dist) / 2

            # if your speed becomes zero because you got stuck... negative
            if c_speed < 0.5 and n_speed < 0.5:
                reward -= time
            elif c_speed < 0.5 and np.abs(n_speed) > 1.0:
                reward += time

            # reward stablization of the puck steering near the goal
            reward_angle = continuous_causal_reward_ext(np.abs(c_pg_angle - c_counter), np.abs(n_pg_angle - n_counter), 0.01, 0.0, MAX_STEERING_ANGLE_REWARD) / MAX_STEERING_ANGLE_REWARD

            reward += reward_angle * (MAX_DISTANCE - c_pp_dist) * (time if reward_angle < 0 else 1/time)

        if c_choice == PlayerPuckGoalPlannerActor.CLASSIFIER_STUCK_AGAINST_WALL:
            # maximize the approach towards the target speed
            reward = continuous_causal_reward_ext(c_speed - c_target_speed, n_speed - n_target_speed, 0.1, 0.0, MAX_SPEED) / MAX_SPEED

        return [reward] * PlayerPuckGoalFineTunedPlannerActorNetwork.OUTPUTS
    
    def extract_greedy_action(self, action, f):        
        # determine the steering direction and speed  
        output = self.action_net(f).detach().numpy()
        return output > 0
        
    def log_prob(self, output, actions):
        # skip the first value of the actions (it contains the planner choice)
        actions = actions[:,1:]
        retval = super().log_prob(output, actions=actions)
        return retval
