#!/bin/python3
# Author: Jose Rojas (jlrojas@utexas.edu)
#
# A wrapper around runner and train for executing repetitve reinforement learning data collection and training
#

from state_agent.runner import main as runner

def load_module_file(package, file_path):
    import importlib

    spec = importlib.util.spec_from_file_location(package, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def loop():

    from argparse import ArgumentParser
    

    parser = ArgumentParser(description="Collect data via runner, run a trainer, then repeat")
    parser.add_argument('agent', help="Actor team to load")
    parser.add_argument('--loops', default=1, type=int, help="Number of training loops")
    parser.add_argument('--trainer', type=str, help="Training model to run")    
    args = parser.parse_known_args()[0]

    trainer_module = load_module_file('state_agent.trainers.train', 'state_agent/trainers/{}.py'.format(args.trainer))
    
    for epoch in range(args.loops):

        # collect data via runner
        runner()

        # run the trainer
        trainer_module.main()

        
if __name__ == '__main__':
    loop()




