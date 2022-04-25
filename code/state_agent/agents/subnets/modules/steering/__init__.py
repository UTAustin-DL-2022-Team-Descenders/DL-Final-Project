# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from json import load
from state_agent.agents.subnets.actors import SteeringActor
from state_agent.agents.subnets.agents import BaseTeam, Agent
from state_agent.agents.subnets.utils import load_model, save_model

class Team(BaseTeam):

    def __init__(self, num_of_players=2, train=False):

        # load network
        print("init agent")

        action_net = load_model('agent.pt', file=__file__)
        self.training_actor = SteeringActor(action_net)

        super().__init__(            
            Agent(
                self.training_actor, 
                target_speed=10.0, train=train
            )            
        )

    def save(self):

        # save network
        save_model(self.training_actor.action_net, 'agent.pt', file=__file__)

    def get_training_actor(self):
        return self.training_actor

    def set_training_mode(self, mode):
        print("Training mode", mode)
        self.agent.train = mode != None        