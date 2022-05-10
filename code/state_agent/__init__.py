# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from typing import List
import torch
import time
import random

if "1.9" not in torch.__version__:
    print("WARNING! Submission grader is using a different torch version than locally installed! Use 1.9.x")

LEGAL_KART_NAMES = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue', 'tux']

# Fix Team's player karts or Agent target speeds. Randomized if not given
# The length of these lists must match num_of_players (== 2)
TEAM_KART_LIST = []
AGENT_TARGET_SPEED = []

class Team():
    agent_type = 'state'
    def __init__(self, num_of_players=2, train=False, time_act_func=False,
            team_kart_list=TEAM_KART_LIST, agent_target_speed_list=AGENT_TARGET_SPEED):
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor
        from .agents import BaseTeam, Agent, BaseAgent


        # From BaseTeam
        self.team = None
        self.num_players = num_of_players
        self.training_mode = None
        self.agents: List[BaseAgent] = []

        # List of target agent speeds
        self.agent_target_speed_list = agent_target_speed_list

        # Add random agent target speeds if one wasn't given for all num_players
        for i in range(self.num_players - len(self.agent_target_speed_list)):
            agent_target_speed = random.uniform(12, 21)
            self.agent_target_speed_list.append(agent_target_speed)

        # List of karts on this team
        self.team_kart_list = team_kart_list

        # Add random karts to team_kart_list if one wasn't given for all num_players
        for i in range(self.num_players - len(self.team_kart_list)):
            kart_name = random.choice(LEGAL_KART_NAMES)
            self.team_kart_list.append(kart_name)

        # Check that team_kart_list are legal kart names
        assert(set(self.team_kart_list).issubset(set(LEGAL_KART_NAMES)))

        # Print information about this Team
        print(f"Team - carts in use:", end=" ")
        for i in range(num_of_players):
            print(f"{self.team_kart_list[i]} using speed {agent_target_speed_list[i]:.1f}", end="; ")
        print("\n", end="")

        # Set flag to denote if we're timing the act function or not
        self.time_act_func = time_act_func

        # set slowest time to act for tracking worst case
        self.slowest_act_time = 0.

    def set_training_mode(self, mode):
        self.training_mode = mode

    def new_match(self, team: int, num_players: int) -> list:
        
        self.team, self.num_players = team, num_players

        use_jit = True

        for agent in self.agents:
            agent.reset()

        if use_jit:
            self.agents = [
                self.create_composed_network("agent_basic_net", # use a different name for agent 1 vs agent 2 based on the configured actors
                    target_speed=self.agent_target_speed_list[0]
                ),
                self.create_composed_network("agent_basic_net",
                    target_speed=self.agent_target_speed_list[1]
                )
            ]
        else:
            self.create_discrete_networks()

        return self.team_kart_list

    def act(self, player_states, opponent_states, soccer_state):
        from .core_utils import DictObj

        # Collect start time if timeit is set
        if self.time_act_func:
            start_time = time.time()

        # actions list for all players on this team
        actions = []

        # Iterate over all players on this team
        for player_num, player_state in enumerate(player_states):
            # Get network output by forward feeding features through the model
            output = dict(vars(self.agents[player_num](DictObj(player_state['kart']), DictObj(soccer_state), self.team)))

            # add action dictionary to actions list
            actions.append(output)

        # Print act execute time if timeit is set
        if self.time_act_func:
            end_time = time.time()
            act_time = end_time-start_time

            # Print only the slowest act executions
            if act_time > self.slowest_act_time:
                self.slowest_act_time = act_time
                print(f'Team.act slowest act in {(act_time*1000):.1f}ms')
            
            # Print act execution every timestep. 
            # WARNING: adds huge number of print statements
            #print(f'Team.act in {(act_time*1000):.1f}ms')


        return actions

    def create_discrete_networks(self):
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor
        from .agents import BaseTeam, Agent

        self.steering_actor = SteeringActor()
        self.steering_actor.load_model(use_jit=True)

        self.speed_actor = SpeedActor()
        self.speed_actor.load_model(use_jit=True)

        self.drift_actor = DriftActor()
        self.drift_actor.load_model(use_jit=True)

        self.planner_actor = PlayerPuckGoalPlannerActor()
        self.planner_actor.load_model(use_jit=True)

        #self.ft_planner_actor = PlayerPuckGoalFineTunedPlannerActor(mode="speed")
        #self.ft_planner_actor.load_model(use_jit=True)


        self.agents = [
            Agent(
                self.planner_actor,
                #self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=self.agent_target_speed_list[0]
            ),
            Agent(
                self.planner_actor,
                # self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=self.agent_target_speed_list[1]
            )
        ]


    def create_composed_network(self, model_name, target_speed=23.0):
        from .agents import BaseTeam, Agent, ComposedAgent, ComposedAgentNetwork
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor

        composed_network = ComposedAgentNetwork(
            SteeringActor().action_net,  # type: ignore
            SpeedActor().actor_net,      # type: ignore
            DriftActor().actor_net,      # type: ignore
            PlayerPuckGoalPlannerActor().actor_net  # type: ignore
            #None   # type: ignore
        )

        agent = ComposedAgent(composed_network, target_speed=target_speed)
        agent.load_models(model_name, use_jit=True)

        return agent

    def save_composed_network(self, model_name):
        from .agents import BaseTeam, Agent, ComposedAgent, ComposedAgentNetwork
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor

        steering_actor = SteeringActor()
        steering_actor.load_model(use_jit=True)

        speed_actor = SpeedActor()
        speed_actor.load_model(use_jit=True)

        drift_actor = DriftActor()
        drift_actor.load_model(use_jit=True)

        planner_actor = PlayerPuckGoalPlannerActor()
        planner_actor.load_model(use_jit=True)

        composed_network = ComposedAgentNetwork(
            steering_actor.actor_net,  # type: ignore
            speed_actor.actor_net,     # type: ignore
            drift_actor.actor_net,     # type: ignore
            planner_actor.actor_net    # type: ignore
            #None   # type: ignore
        )

        agent = ComposedAgent(composed_network)
        agent.save_models(model_name, use_jit=True)
        print(agent.load_models(model_name, use_jit=True).state_dict())

        return agent
