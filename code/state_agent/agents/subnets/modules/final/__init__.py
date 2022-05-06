# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from state_agent.agents.subnets.actors import SteeringActor, SpeedActor
from state_agent.agents.subnets.planners import PlayerPuckGoalPlannerActor
from state_agent.agents.subnets.agents import BaseTeam, Agent

class Team(BaseTeam):

    def __init__(self, num_of_players=2, train=False):


        self.steering_actor = SteeringActor()
        self.steering_actor.load_model(use_jit=False)

        self.speed_actor = SpeedActor()
        self.speed_actor.load_model(use_jit=False)

        self.training_actor = PlayerPuckGoalPlannerActor(self.speed_actor, self.steering_actor)
        self.training_actor.load_model(use_jit=False)

        agent = Agent(self.training_actor, train=train)

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