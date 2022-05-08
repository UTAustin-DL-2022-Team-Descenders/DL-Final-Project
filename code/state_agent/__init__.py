# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

import torch
#import numpy as np
if "1.9" not in torch.__version__:
    print("WARNING! Submission grader is using a different torch version than locally installed! Use 1.9.x")
class Team():
    agent_type = 'state'
    def __init__(self, num_of_players=2, train=False):
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor

        # From BaseTeam
        self.team = None
        self.num_players = 0
        self.training_mode = None

        self.steering_actor = SteeringActor()
        self.steering_actor.load_model(use_jit=True)

        self.speed_actor = SpeedActor()
        self.speed_actor.load_model(use_jit=True)

        self.drift_actor = DriftActor()
        self.drift_actor.load_model(use_jit=True)

        # TODO: how are planners initiated now?
        self.planner_actor = PlayerPuckGoalPlannerActor()
        self.planner_actor.load_model(use_jit=True)

        self.ft_planner_actor = PlayerPuckGoalFineTunedPlannerActor(mode="speed")
        self.ft_planner_actor.load_model(use_jit=True)



    def set_training_mode(self, mode):
        self.training_mode = mode

    def new_match(self, team: int, num_players: int) -> list:

        from .agents import BaseTeam, Agent

        carts = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue']

        self.team, self.num_players = team, num_players

        # TODO: how is the Agent initiated now?

        #speeds = [np.random.uniform(17.0, 23.0), np.random.uniform(12.0, 16.0)]
        self.agents = [
            Agent(
                self.planner_actor,
                self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=19.0 #speeds[0]
            ),
            Agent(
                self.planner_actor,
                self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=15.5 #speeds[1]
            )
        ]

        for agent in self.agents:
            agent.reset()

        #indx = np.random.uniform(low=0, high=len(carts), size=[2]).astype(np.int8)

        #cart_names = []
        #cart_names.append(carts[indx[0]])
        #cart_names.append(carts[indx[1]])

        #print(cart_names)
        #print(speeds)

        return ['konqi', 'hexley']

    def act(self, player_states, opponent_states, soccer_state):
        from .core_utils import DictObj
        # actions list for all players on this team
        actions = []

        # Iterate over all players on this team
        for player_num, player_state in enumerate(player_states):
            # Get network output by forward feeding features through the model
            output = self.agents[player_num](DictObj(player_state['kart']), DictObj(soccer_state), self.team)

            # add action dictionary to actions list
            actions.append(dict(vars(output)))

        return actions