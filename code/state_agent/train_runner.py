#!/bin/python3
# Author: Jose Rojas (jlrojas@utexas.edu)
#
# A wrapper around runner and train for executing repetitve reinforement learning data collection and training
#

from argparse import Namespace
from state_agent.runner import main as runner_main, runner
from state_agent.utils import load_module_file

def loop():

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Collect data via runner, run a trainer, then repeat")
    parser.add_argument('agent', help="Actor team to load")
    parser.add_argument('--epochs', default=1, type=int, help="Number of training epochs")
    parser.add_argument('--trainer', type=str, help="Training model to run")    
    args = parser.parse_known_args()[0]

    trainer_module = load_module_file('state_agent.trainers.train', 'state_agent/trainers/{}.py'.format(args.trainer))
    
    for epoch in range(args.epochs):

        # collect data via runner using it's CLI
        teams, trajectories = runner_main(Namespace(
            training_mode=trainer_module.__class__.__name__
        ))

        # run the trainer with the runner as the rollout function for validation        
        args = Namespace(
            rollout=runner_main,
            comparison_actor=teams[1],
            actor=teams[0], # XXX assumes the agent is on the first team!
            trajectories=trajectories
        )
        trainer_module.main(args)

        
if __name__ == '__main__':
    loop()




