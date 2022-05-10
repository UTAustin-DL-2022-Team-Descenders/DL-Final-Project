# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from typing import Any, List, Tuple
import torch
import copy
import os.path
from functools import reduce
from state_agent.planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor
from state_agent.actors import BaseActor
from state_agent.actors import DriftActorNetwork, SpeedActorNetwork, SteeringActorNetwork
from state_agent.core_utils import save_model, load_model
from state_agent.actors import BaseActorNetwork, Action
from state_agent.features import SoccerFeatures, MAX_SPEED, extract_all_features
from state_agent.core_utils import DictObj

class BaseAgent:

    MAX_STATE = 5

    def __init__(self, *args, extractor:Any=None, train=False, target_speed=None, **kwargs):
        self.actors = list(args)
        self.train = train
        self.extractor = extractor
        self.target_speed = target_speed
        self.reset()
    
    def invoke_actor(self, actor, action, f):
        x = torch.as_tensor(actor.select_features(f)).view(-1)
        actor(action, x, train=self.train, extractor=f)

    def invoke_actors(self, action, f):
        [self.invoke_actor(actor, action, f) for actor in self.actors]
        return action

    def get_feature_vector(self, kart_info, soccer_state, team_num, **kwargs):
        return self.extractor(
            kart_info, 
            soccer_state,
            team_num,
            target_speed=self.target_speed,            
            **kwargs
        )

    def reset(self):
        self.last_output = None
        self.last_state = []

    def __call__(self, kart_info, soccer_state, team_num, **kwargs):
        action = Action()

        f = self.get_feature_vector(kart_info, soccer_state, team_num, last_state=self.last_state, last_action=self.last_output)

        # save previous kart state
        self.last_state.append(copy.deepcopy(kart_info))
        if len(self.last_state) > self.MAX_STATE:
            self.last_state.pop(0)

        action = self.invoke_actors(action, f)

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

# NOTE: This Agent is used for training purposes. It has some utilities for post processing the acceleration for example.
# It is not used for the final evaluations.
class Agent(BaseAgent):
    def __init__(self, *args, target_speed=MAX_SPEED, **kwargs):
        super().__init__(*args, extractor=extract_all_features, target_speed=target_speed, **kwargs)

        self.accel = torch.tensor(kwargs['accel'] if 'accel' in kwargs else 1.0)
        self.use_accel = not reduce(lambda x, y: x or hasattr(y, "acceleration"), self.actors, False)


    def invoke_actors(self, action, f):
        super().invoke_actors(action, f)

        # NOTE: this is only called for TRAINING agents, the Evaluation agent will not post-process!
        if self.use_accel:
            action.acceleration = self.accel

        return action

class ComposedAgentNetwork(torch.nn.Module):

    """
    The composited agent network is a torch module that runs a sequence of BaseActorNetworks
    """

    def __init__(self,
    steering_actor: SteeringActorNetwork,
    speed_actor: SpeedActorNetwork,
    drift_actor: DriftActorNetwork,
    planner_actor: PlayerPuckGoalPlannerActor
    ) -> None:
        super().__init__()

        self.steering_actor = steering_actor
        self.speed_actor = speed_actor
        self.drift_actor = drift_actor
        self.planner_actor = planner_actor
        #self.ft_planner_actor = ft_planner_actor


    def forward(self, action: Action, f: SoccerFeatures):

        # the struggle is real with TorchScript.. you can't call helper methods that use nn.Module so these lines have to be duplicated

        if self.planner_actor is not None:
            x = torch.as_tensor(self.planner_actor.select_features(f)).view(-1)
            self.planner_actor(action, x, f)

        if self.steering_actor is not None:
            x = torch.as_tensor(self.steering_actor.select_features(f)).view(-1)
            self.steering_actor(action, x, f)

        if self.speed_actor is not None:
            x = torch.as_tensor(self.speed_actor.select_features(f)).view(-1)
            self.speed_actor(action, x, f)

        if self.drift_actor is not None:
            x = torch.as_tensor(self.drift_actor.select_features(f)).view(-1)
            self.drift_actor(action, x, f)

        return action

class ComposedAgent(BaseAgent):

    """
    # NOTE: The composed agent is agent used for evaluations and loads a single agent network of actors networks, for the purpose
    #  of executing them all as one torch module
    """

    def __init__(self, agent_net: ComposedAgentNetwork = None, target_speed=MAX_SPEED, **kwargs):  # type: ignore
        super().__init__(*[], extractor=extract_all_features, target_speed=target_speed, **kwargs)
        self.agent_net = agent_net if agent_net else ComposedAgentNetwork(
            SteeringActorNetwork(),
            SpeedActorNetwork(),
            DriftActorNetwork(),
            PlayerPuckGoalPlannerActor()
        )

    def save_models(self, model_name, use_jit=False):
        save_model(self.agent_net, model_name, save_path=os.path.abspath(os.path.dirname(__file__)), use_jit=use_jit)

    def load_models(self, model_name, use_jit=False):
        self.agent_net = load_model(model_name, model=self.agent_net, load_path=os.path.abspath(os.path.dirname(__file__)), use_jit=use_jit)
        return self.agent_net

    def invoke_actors(self, action: Action, f: SoccerFeatures):
        # return the action from the net directly
        return self.agent_net(action, f)

class BaseTeam:
    agent_type = 'state'

    def __init__(self, agents):
        self.team = None
        self.num_players = 0
        self.training_mode = None
        self.agents = agents

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
        
        for agent in self.agents:
            agent.reset()
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
          output = self.agents[player_num](DictObj(player_state['kart']), DictObj(soccer_state), self.team)

          # add action dictionary to actions list
          actions.append(dict(vars(output)))

        return actions


