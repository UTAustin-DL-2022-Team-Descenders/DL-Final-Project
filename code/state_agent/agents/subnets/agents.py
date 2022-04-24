# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from argparse import Namespace
import torch
from functools import reduce
from .features import SoccerFeatures
from .utils import DictObj

class Action:
    
    def __init__(self):
        self.acceleration = 0.0
        self.steer = 0.0
        self.drift = False
        self.nitro = False
        self.brake = False
        self.fire = False

class BaseAgent:
    def __init__(self, *args, extractor=None, train=False, target_speed=None, **kwargs):
        self.nets = args
        self.train = train
        self.extractor = extractor
        self.accel = kwargs['accel'] if 'accel' in kwargs else 1.0
        self.use_accel = not reduce(lambda x, y: x or hasattr(y, "acceleration"), self.nets, False)
        self.target_speed = target_speed
        self.last_output = torch.Tensor([0, 0, 0, 0, 0])        
    
    def invoke_actor(self, actor, action, f):
        actor(action, actor.select_features(self.extractor, f), train=self.train)       

    def invoke_actors(self, action, f):
        [self.invoke_actor(actor, action, f) for actor in self.nets]

    def get_feature_vector(self, kart_info, soccer_state, **kwargs):
        return self.extractor.get_feature_vector(kart_info, soccer_state, target_speed=self.target_speed, **kwargs)

    def __call__(self, player_num, kart_info, soccer_state, **kwargs):
        action = Action()
        action.acceleration = self.accel

        f = self.get_feature_vector(kart_info, soccer_state)
        f = torch.as_tensor(f).view(-1)
        self.invoke_actors(action, f) 
        if self.use_accel:
            action.acceleration = self.accel       
        return action

class Agent(BaseAgent):
    def __init__(self, *args, target_speed=10.0, **kwargs):
        super().__init__(*args, extractor=SoccerFeatures(), target_speed=target_speed, **kwargs)

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
          output = self.agent(player_num, DictObj(player_state['kart']), DictObj(soccer_state))

          # add action dictionary to actions list
          actions.append(dict(vars(output)))

        return actions

