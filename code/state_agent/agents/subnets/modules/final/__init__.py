# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from state_agent.agents.subnets.actors import SteeringActor, SpeedActor, DriftActor
from state_agent.agents.subnets.planners import PlayerPuckGoalPlannerActor
from state_agent.agents.subnets.agents import BaseTeam, Agent

class Team(BaseTeam):

    def __init__(self, num_of_players=2, train=False):

        self.steering_actor = SteeringActor()
        self.steering_actor.load_model(use_jit=False)

        self.speed_actor = SpeedActor()
        self.speed_actor.load_model(use_jit=False)

        self.drift_actor = DriftActor()
        self.drift_actor.load_model(use_jit=False)

        # TODO: how are planners initiated now?
        self.planner_actor = PlayerPuckGoalPlannerActor()
        self.planner_actor.load_model(use_jit=False)

        # TODO: how is the Agent initiated now?
        # agent = Agent(self.training_actor, train=train)
        agent = Agent(
            self.planner_actor,
            self.steering_actor,
            self.speed_actor,
            self.drift_actor,
            target_speed=15.0
        )

        super().__init__(
            agent
        )

    def save(self):
        self.agent.save_models()


    def get_training_actor(self):
        return self.training_actor

    def set_training_mode(self, mode):
        print("Training mode", mode)
        self.agent.train = mode != None