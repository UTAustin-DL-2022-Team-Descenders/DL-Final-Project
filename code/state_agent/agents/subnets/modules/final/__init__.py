# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from state_agent.agents.subnets.actors import SteeringActor, SpeedActor, DriftActor
from state_agent.agents.subnets.planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor
from state_agent.agents.subnets.agents import BaseTeam, Agent

class Team(BaseTeam):

    def __init__(self, num_of_players=2, train=False):

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

        # TODO: how is the Agent initiated now?
        agents = [
            Agent(
                self.planner_actor,
                self.ft_planner_actor,
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

        super().__init__(
            agents
        )
