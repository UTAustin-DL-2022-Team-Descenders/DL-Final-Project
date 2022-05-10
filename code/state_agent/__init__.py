# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from typing import List
import torch

if "1.9" not in torch.__version__:
    print("WARNING! Submission grader is using a different torch version than locally installed! Use 1.9.x")

class Team():
    agent_type = 'state'
    def __init__(self, num_of_players=2, train=False):
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor
        from .agents import BaseTeam, Agent, BaseAgent

        # From BaseTeam
        self.team = None
        self.num_players = 0
        self.training_mode = None
        self.agents: List[BaseAgent] = []

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
                    target_speed=21.0
                ),
                self.create_composed_network("agent_basic_net",
                    target_speed=12.0
                )
            ]
        else:
            self.create_discrete_networks()

        return ['tux'] * num_players

    def act(self, player_states, opponent_states, soccer_state):
        from .core_utils import DictObj
        # actions list for all players on this team
        actions = []

        # Iterate over all players on this team
        for player_num, player_state in enumerate(player_states):
            # Get network output by forward feeding features through the model
            output = dict(vars(self.agents[player_num](DictObj(player_state['kart']), DictObj(soccer_state), self.team)))

            # add action dictionary to actions list
            actions.append(output)

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
                target_speed=21.0
            ),
            Agent(
                self.planner_actor,
                # self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=12.0
            )
        ]


    def create_composed_network(self, model_name, target_speed=23.0):
        from .agents import BaseTeam, Agent, ComposedAgent, ComposedAgentNetwork
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor

        composed_network = ComposedAgentNetwork(
            SteeringActor().actor_net,
            SpeedActor().actor_net,
            DriftActor().actor_net,
            PlayerPuckGoalPlannerActor().actor_net
            #None   # type: ignore
        )

        agent = ComposedAgent(composed_network, target_speed=target_speed)
        agent.load_models(model_name, use_jit=True)

        return agent

    def save_composed_network(self, model_name):
        from .agents import BaseTeam, Agent, ComposedAgent, ComposedAgentNetwork
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor

        planner_actor = PlayerPuckGoalPlannerActor()
        planner_actor.load_model(use_jit=True)

        steering_actor = SteeringActor()
        steering_actor.load_model(use_jit=True)

        speed_actor = SpeedActor()
        speed_actor.load_model(use_jit=True)

        drift_actor = DriftActor()
        drift_actor.load_model(use_jit=True)

        composed_network = ComposedAgentNetwork(
            steering_actor.actor_net,
            speed_actor.actor_net,
            drift_actor.actor_net,
            planner_actor.actor_net
            #None   # type: ignore
        )

        agent = ComposedAgent(composed_network)
        agent.save_models(model_name, use_jit=True)
        print(agent.load_models(model_name, use_jit=True).state_dict())

        return agent
