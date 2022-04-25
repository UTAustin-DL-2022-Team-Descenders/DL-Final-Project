#!/bin/python3
# Author: Jose Rojas (jlrojas@utexas.edu)
#
# A wrapper around runner and train for executing repetitve reinforement learning data collection and training
#
import time

from argparse import Namespace
from state_agent.runner import main as runner_main
from state_agent.utils import load_module_file

def train(trainer_module, epoch, context=None):
    # collect data via runner using it's CLI

    teams, trajectories = runner_main(Namespace(
        training_mode=trainer_module.__class__.__name__
    ))
    
    # run the trainer
    context = trainer_module.train(
        epoch=epoch,            
        team=teams[0], # XXX assumes the agent is on the first team!
        trajectories=trajectories,
        context=context
    )

    return context, teams
    

def validate(trainer_module, epoch, context=None, teams=[]):
    # run validation phase
    teams, trajectories = runner_main()
    
    # run the validation
    return trainer_module.validate(
        epoch=epoch,            
        team=teams[0], # XXX assumes the agent is on the first team!
        trajectories=trajectories,
        context=context
    )

def loop(
    epochs=10,
    trainer:str="",
    validation="on_startup",
    **kwargs
):

    trainer_module = load_module_file('state_agent.trainers.train', 'state_agent/trainers/{}.py'.format(trainer))
    
    # run an initial validation phase to determine model performance    
    context = None
    if validation == "on_startup":
        # run the validation
        context = validate(trainer_module, 0)
        
    for epoch in range(epochs):

        context, teams = train(trainer_module, epoch, context)
        if validation:
            validate(trainer_module, epoch, context, teams)

        
        
if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Collect data via runner, run a trainer, then repeat")
    parser.add_argument('agent', help="Actor team to load")
    parser.add_argument('--epochs', default=1, type=int, help="Number of training epochs")
    parser.add_argument('--trainer', type=str, help="Training model to run")    
    args = parser.parse_known_args()[0]

    loop(**vars(args))




