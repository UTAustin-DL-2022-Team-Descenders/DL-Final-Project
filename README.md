# DL-Final-Project

## Example train_reinforce run commands.
Note that train_reinforce trajectory data is generated and used in code/state_agent/reinforce_data.

Create a new ActionNetwork and roll out 3 trajectories per 2 epochs

    python -m state_agent.trainers.train_reinforce -nt 3 -ep 2

Load an existing ActionNetwork from code/state_agent/state_agent.pt and rolls out against only jurgen_agent. Perform training using SGD and learning rate of 0.01.

    python -m state_agent.trainers.train_reinforce --load_model --training_opponent jurgen_agent --optimizer SGD -lr .01

For a complete list of available command line options.

    python -m state_agent.trainers.train_reinforce --help

## Example train_imitation run command (**After generating imitation data in code/state_agent/imitation_data**):

Create a new ActionNetwork and performs imitation learning for 10 epochs.

    python -m state_agent.trainers.train_imitation -ep 10

Load an existing ActionNetwork from code/state_agent/state_agent.pt and performs imitation learning.

    python -m state_agent.trainers.train_imitation --load_model

For a complete list of available command line options.

    python -m state_agent.trainers.train_imitation --help
