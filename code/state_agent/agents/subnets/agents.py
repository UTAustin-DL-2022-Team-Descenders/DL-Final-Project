# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from argparse import Namespace
from typing import Union
import torch
import copy
from functools import reduce
from state_agent.agents.subnets.features import SoccerFeatures, MAX_SPEED
from state_agent.agents.subnets.utils import DictObj

class Action:

    def __init__(self):
        self.acceleration = torch.Tensor([0.0])
        self.steer = torch.Tensor([0.0])
        self.drift = torch.Tensor([False])
        self.nitro = torch.Tensor([False])
        self.brake = torch.Tensor([False])
        self.fire = torch.Tensor([False])

    def detach(self):
        self.acceleration = float(self.acceleration.detach().numpy()) if hasattr(self.acceleration, "detach") else self.acceleration
        self.steer = float(self.steer.detach().numpy()) if hasattr(self.steer, "detach") else self.steer
        self.drift = bool(self.drift.detach().numpy()) if hasattr(self.drift, "detach") else self.drift
        self.nitro = bool(self.nitro.detach().numpy()) if hasattr(self.nitro, "detach") else self.nitro
        self.brake = bool(self.brake.detach().numpy()) if hasattr(self.brake, "detach") else self.brake
        self.fire = bool(self.fire.detach().numpy()) if hasattr(self.fire, "detach") else self.fire
class BaseAgent:

    MAX_STATE = 5

    def __init__(self, *args, extractor=None, train=False, target_speed=None, **kwargs):
        self.actors = args
        self.train = train
        self.extractor = extractor
        self.accel = kwargs['accel'] if 'accel' in kwargs else 1.0
        self.use_accel = not reduce(lambda x, y: x or hasattr(y, "acceleration"), self.actors, False)
        self.target_speed = target_speed
        self.reset()
    
    def invoke_actor(self, actor, action, f):
        actor(action, torch.as_tensor(actor.select_features(f)).view(-1), train=self.train, extractor=f)

    def invoke_actors(self, action, f):
        [self.invoke_actor(actor, action, f) for actor in self.actors]

    def get_feature_vector(self, kart_info, soccer_state, **kwargs):        
        return self.extractor(
            kart_info, 
            soccer_state, 
            target_speed=self.target_speed,            
            **kwargs
        )

    def reset(self):
        self.last_output = None
        self.last_state = []

    def __call__(self, kart_info, soccer_state, **kwargs):
        action = Action()
        action.acceleration = self.accel

        f = self.get_feature_vector(kart_info, soccer_state, last_state=self.last_state, last_action=self.last_output)

        # save previous kart state
        self.last_state.append(copy.deepcopy(kart_info))
        if len(self.last_state) > self.MAX_STATE:
            self.last_state.pop(0)

        self.invoke_actors(action, f)         
        if self.use_accel:
            action.acceleration = self.accel

        action.detach()
        self.last_output=action
        
        return action

    def save_models(self):
        for actor_i in range(len(self.actors)):
            actor = self.actors[actor_i]
            actor_model_save_name = f"{self.actors[actor_i].model_name}_{actor_i}"
            actor.save_model(actor_model_save_name)

    def load_models(self):
        for actor_i in range(len(self.actors)):
            actor = self.actors[actor_i]
            actor_model_load_name = f"{self.actors[actor_i].model_name}_{actor_i}"
            actor.load_model(actor_model_load_name)

class Agent(BaseAgent):
    def __init__(self, *args, target_speed=MAX_SPEED, **kwargs):
        super().__init__(*args, extractor=SoccerFeatures, target_speed=target_speed, **kwargs)

class BaseTeam:
    agent_type = 'state'

    def __init__(self, agent: Agent):
        self.team = None
        self.num_players = 0
        self.training_mode = None
        self.agent = agent

    def set_training_mode(self, mode):
        """
        The training mode algorithm will be passed by name here.
        This allows the agent to decide what actions to take based on the training type.
        For example, a "reinforce" mode could use randomized actions from a policy distribution
        """
        self.training_mode = mode

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """

        self.team, self.num_players = team, num_players
        
        self.agent.reset()
        return ['tux'] * num_players

    def act(self, player_states, opponent_states, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_states: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_states: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # actions list for all players on this team
        actions = []

        # Iterate over all players on this team
        for player_num, player_state in enumerate(player_states):

          # Get network output by forward feeding features through the model
          output = self.agent(DictObj(player_state['kart']), DictObj(soccer_state))

          # add action dictionary to actions list
          actions.append(dict(vars(output)))

        return actions


