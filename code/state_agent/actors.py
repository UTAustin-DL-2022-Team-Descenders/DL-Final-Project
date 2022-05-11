# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import torch
from torch.jit import script as torchscript
from torch.distributions import Bernoulli, Normal, Categorical, OneHotCategorical
from state_agent.action_nets import LinearNetwork
from state_agent.action_nets import BaseNetwork
from state_agent.features import SoccerFeatures
from state_agent.action_nets import LinearWithTanh
from state_agent.rewards import steering_angle_reward, speed_reward
from state_agent.core_utils import save_model, load_model
import os

@torchscript
class Action:

    def __init__(self):
        self.acceleration = torch.tensor([0.0])
        self.steer = torch.tensor([0.0])
        self.drift = torch.tensor([False])
        self.nitro = torch.tensor([False])
        self.brake = torch.tensor([False])
        self.fire = torch.tensor([False])

    def detach(self):
        self.acceleration.detach()
        self.steer.detach()
        self.drift.detach()
        self.nitro.detach()
        self.brake.detach()
        self.fire.detach()

class BaseActorNetwork(torch.nn.Module):

    def __init__(self, action_net: torch.nn.Module) -> None:
        super().__init__()
        self.action_net = action_net

    def select_features(self, features: SoccerFeatures):
        # this is only called for top level actors; nested actors are given features directly from their ancestors
        pass

    def forward(self, x):
        pass

class BaseActor:

    def __init__(self, actor_net: BaseActorNetwork, train=None, reward_type=None, sample_type=None):
        self.actor_net = actor_net.cpu().eval() if train != True else actor_net
        self.train = train
        self.reward_type = reward_type
        self.sample_type = sample_type

        # Set model path and name to save/load BaseActor's action_net
        self.model_name = "base_actor"

    @property
    def action_net(self) -> LinearNetwork:
        return self.actor_net.action_net

    def copy(self, actor_net):
        return self.__class__(actor_net, train=self.train, reward_type=self.reward_type)

    def __call__(self, action, f, train=False, **kwargs):        
        output = self.actor_net(f)
        if self.train is not None:
            train = self.train
        #assert(self.action_net.training == train)            
        if train:       
            # choose a set of labels by sampling
            output = self.sample(output)
        return output

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
        if  type(self.actor_net.action_net) == LinearNetwork and \
            self.action_net.activation == "Tanh" or \
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


    def save_model(self, custom_model_name=None, use_jit=False):

        # set the save name of the model. User may provide a custom model name or default to self.model_name
        save_model_name = custom_model_name if custom_model_name else self.model_name

        save_model(self.actor_net, save_model_name, save_path=os.path.abspath(os.path.dirname(__file__)), use_jit=use_jit)

        return self.actor_net

    def load_model(self, custom_model_name=None, model=None, use_jit=False, conversion=None):

        model = model if model else self.actor_net

        # set the load name of the model. User may provide a custom model name or default to self.model_name
        load_model_name = custom_model_name if custom_model_name else self.model_name

        self.actor_net = load_model(load_model_name, load_path=os.path.abspath(os.path.dirname(__file__)), model=model, use_jit=use_jit, conversion=conversion)

        return self.actor_net

    def load_action_net(self, custom_model_name=None, model=None, use_jit=False, conversion=None):

        model = model if model else self.actor_net.action_net

        # set the load name of the model. User may provide a custom model name or default to self.model_name
        load_model_name = custom_model_name if custom_model_name else self.model_name

        self.actor_net.action_net = load_model(load_model_name, load_path=os.path.abspath(os.path.dirname(__file__)), model=model, use_jit=use_jit, conversion=conversion)

        return self.actor_net.action_net

    def select_features(self, features: SoccerFeatures):
        return self.actor_net.select_features(features)

class SteeringActorNetwork(BaseActorNetwork):

    def __init__(self):
        # Steering action_net
        # inputs: delta steering angle
        super().__init__(LinearWithTanh(1, 1, n_hidden=20, bias=False, scale=None, range=None))

    def forward(self, action: Action, f: torch.Tensor, features: SoccerFeatures):
        output = self.action_net(f)
        action.steer = output[0]
        return output

    def select_features(self, features: SoccerFeatures):
        delta_steering_angle = features.select_steering_angle()
        return delta_steering_angle[None]

class SteeringActor(BaseActor):

    def __init__(self, actor_net=None, train=None, **kwargs):

        # Steering action_net
        # inputs: delta steering angle
        super().__init__(actor_net if actor_net else SteeringActorNetwork(), train=train, sample_type="bernoulli")

        # Set model path to save/load SteeringActor's action_net
        self.model_path = os.path.abspath(os.path.dirname(__file__))

        # Set model name for saving and loading action net
        self.model_name = "steer_net"


    def __call__(self, action, f, extractor, train=False, **kwargs):
        output = self.actor_net(action, f, extractor)
        if self.train is not None:
            train = self.train
        if train:
            action.steer = self.sample(output) * 2 - 1
        return action

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next, time):
        current_angle = selected_features_curr
        next_angle = selected_features_next
        return steering_angle_reward(current_angle, next_angle)

    def extract_greedy_action(self, action, *args, **kwargs):
        return [action.steer > 0]

class DriftActorNetwork(BaseActorNetwork):

    def __init__(self):
        # Steering action_net
        # inputs: delta steering angle
        super().__init__(LinearWithTanh(2, 1, n_hidden=20, bias=True, scale=None, range=None))

    def forward(self, action: Action, f: torch.Tensor, features: SoccerFeatures):
        output = self.action_net(f)
        # drift is a binary value
        action.drift = output[0] > 0
        return output

    def select_features(self, features: SoccerFeatures):
        return features.select_indicies(
            [
                features.STEERING_ANGLE,
                features.PREVIOUS_STEER
            ]
        )

class DriftActor(BaseActor):

    def __init__(self, actor_net=None, train=None, **kwargs):
        # Steering action_net
        # inputs: delta steering angle
        super().__init__(actor_net if actor_net else DriftActorNetwork(), train=train, sample_type="bernoulli")

        # Set model path to save/load DriftActor's action_net
        self.model_path = os.path.abspath(os.path.dirname(__file__))

        # Set model name for saving and loading action net
        self.model_name = "drift_net"


    def __call__(self, action, f, extractor, train=True, **kwargs):
        output = self.actor_net(action, f, extractor)
        if self.train is not None:
            train = self.train
        if train:            
            action.drift = self.sample(output) > 0
        return action

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next, time):
        [current_angle, c_steer] = selected_features_curr
        [next_angle, n_steer] = selected_features_next
        reward = steering_angle_reward(current_angle, next_angle)
        
        # consider drifting only when steering angle is maxed and when the target angle is large
        if torch.abs(c_steer) > 0.99 and torch.abs(next_angle) > 0.25:
            return reward if action.drift == True else -1
        else:
            return reward if action.drift == False else -1 

    def extract_greedy_action(self, action, *args, **kwargs):
        return [action.drift > 0]

class SpeedActorNetwork(BaseActorNetwork):

    def __init__(self):
        # inputs:
        #   delta steering angle
        #   delta speed
        #   target speed
        # outputs:
        #   acceleration (0-1)
        #   brake (boolean)
        super().__init__(LinearWithTanh(3, 2,  n_hidden=20, bias=False, scale=None, range=None))

    def forward(self, action: Action, f: torch.Tensor, features: SoccerFeatures):
        output = self.action_net(f)

        # round output due to continuous gradient never being exactly zero
        action.acceleration = torch.clamp(output[0], 0, 1.0)
        # brake is a binary value
        action.brake = output[1] > 0.5

        return output

    def select_features(self, features: SoccerFeatures):
        delta_steering_angle = features.select_steering_angle()
        delta_speed = features.select_delta_speed()
        target_speed = features.select_target_speed()
        return torch.cat([
            delta_steering_angle[None],
            delta_speed[None],
            target_speed[None]
        ], dim=0)


class SpeedActor(BaseActor):

    acceleration = True

    def __init__(self, actor_net=None, train=None, **kwargs):
        super().__init__(actor_net if actor_net else SpeedActorNetwork(), train=train, sample_type="bernoulli", **kwargs)

        # Set model path to save/load SpeedActor's action_net
        self.model_path = os.path.abspath(os.path.dirname(__file__))

        # Set model name for saving and loading action net
        self.model_name = "speed_net"


    def __call__(self, action, f, extractor, train=True, **kwargs):
        output = self.actor_net(action, f, extractor)
        if self.train is not None:
            train = self.train
        if train:
            sample = self.sample(output)
            action.acceleration = torch.clamp(sample[0], 0, 1.0)
            action.brake = sample[1] > 0.5

        return action

    def reward(self, action, greedy_action, selected_features_curr, selected_features_next, time):

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
