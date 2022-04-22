# DL-Final-Project

## Example train_reinforce run commands.
Note that train_reinforce trajectory data is generated and used in code/state_agent/reinforce_data.

Create a new ActionNetwork and roll out 3 trajectories per 2 epochs

    python -m state_agent.trainers.train_reinforce -nt 3 -ep 2

Load an existing ActionNetwork from code/state_agent/state_agent.pt and rolls out against only jurgen_agent. Perform training using SGD and learning rate of 0.01.

    python -m state_agent.trainers.train_reinforce --load_model --training_opponent jurgen_agent --optimizer SGD -lr .01

For a complete list of available command line options.

    python -m state_agent.trainers.train_reinforce --help

Load an existing ActionNetwork from code/state_agent/state_agent.pt and perform self-play (i.e. play against state_agent). Record a video every 5 games.

    python -m state_agent.trainers.train_reinforce --load_model --training_opponent state_agent --record_video_cadence 5

## Example Imitation Data Generation Command.
Rollout 10 trajectories and save the state pkl files in state_agent/imitation_data. Team1 is set to jurgen_agent and team2 is randomized each trajectory (either jurgen_agent, geoffrey_agent, yann_agent, or yoshua_agent):

    python -m state_agent.utils --datapath state_agent/imitation_data --n_trajectories 10 --team1 jurgen_agent --team2 random

## Example train_imitation run command (**After generating imitation data**):

Create a new ActionNetwork and performs imitation learning for 10 epochs.

    python -m state_agent.trainers.train_imitation -ep 10 -dp state_agent/imitation_data

Load an existing ActionNetwork from code/state_agent/state_agent.pt and performs imitation learning. State_Agent will imitate Team2.

    python -m state_agent.trainers.train_imitation --load_model -dp state_agent/imitation_data --imitation_team 2

For a complete list of available command line options.

    python -m state_agent.trainers.train_imitation --help
