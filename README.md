# DL-Final-Project

Example Reinforce run command:

Creates a new StateAgent and rolls out 3 trajectories (default) over 2 epochs (default)
python -m state_agent.trainers.train_reinforce

Loads an existing StateAgent from state_agent/state_agent.pt and rolls out against only jurgen_agent
python -m state_agent.trainers.train_reinforce --load_model --training_opponent jurgen_agent

For a complete list of available command line options
python -m state_agent.trainers.train_reinforce --help

Example Imitation training run command *After generating imitation data*:

Creates a new ActionNetwork and performs imitation learning with 10 epochs
python -m state_agent.trainers.train_imitation -ep 10

Loads an existing ActionNetwork from state_agent/state_agent.pt and performs imitation learning
python -m state_agent.trainers.train_imitation --load_model
