import subprocess
import torch
import torch.utils.tensorboard as tb
import os, random
from ..utils import *
from ..state_agent import get_features, get_reward, get_score, StateAgent
from .. import save_model

LOGDIR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logdir')
TRAINING_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'reinforce_data')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
RECORD_VIDEO_CADENCE = 1

# Training Knobs

def train_reinforce(args):

    # Base case: negative goals scored (should always do better)
    best_score = -1

    # Create StateAgent
    state_agent = StateAgent(load_existing_model=args.load_model, 
                        optimizer=args.optimizer, 
                        lr=args.learning_rate,
                        gamma=args.gamma
                        )

    #for _ in range(args.epochs):
    while True:

        # Set training opponent so it's visible to end of game results print
        training_opponent = args.training_opponent if args.training_opponent != "random" else get_random_opponent()

        # Set Agent team number. Randomized if args.agent_team == 0
        agent_team_num = args.agent_team if args.agent_team != 0 else random.choice([1,2])

        # Clean out old pkl files and rollout a number of games 
        # and save recordings into pickle files
        clean_pkl_files_and_rollout_many(args.n_trajectories, 
                                        training_opponent=training_opponent, 
                                        agent_team_num=agent_team_num,
                                        output_dir=TRAINING_PATH)

        # Periodically record video to see how the agent is doing
        if state_agent.n_games > 0 and state_agent.n_games % RECORD_VIDEO_CADENCE == 0:
            rollout("state_agent", "state_agent", os.path.abspath(os.path.dirname(__file__)), record_state=False, record_video=True, iteration=state_agent.n_games)
            print("Game %0d recorded" % state_agent.n_games)

        training_pkl_file_list = get_pickle_files(TRAINING_PATH)
        
        # Iterate over game trajectories
        for pkl_file in training_pkl_file_list:

            # Initialize prev_state to None
            prev_state = None

            # Iterate over each timestep in a game recording
            # State_data_dictionaries keys includes team1_state, team2_state, actions, soccer_state
            for state_data_dictionaries in load_recording(pkl_file):

                # Get agent & opponent dictionary key info
                agent_team_key = "team%0d_state" % agent_team_num
                opponent_team_num = 2 if agent_team_num == 1 else 1
                opponent_team_key = "team%0d_state" % opponent_team_num

                # Get player & opponent state dictionaries
                player_states = state_data_dictionaries[agent_team_key]
                opponent_states = state_data_dictionaries[opponent_team_key]

                # Get features from dictionaries to feed into network
                # REVISIT: For now, only train on the same single kart on the team
                player_features = get_features(player_states[0], player_states[1:], 
                                                opponent_states, state_data_dictionaries["soccer_state"],
                                                agent_team_num)
                
                # Use CUDA if available to speed up training
                player_features = player_features.to(DEVICE)

                # Forward pass input through model to get prediction
                prediction_actions = state_agent.get_action_tensor(player_features)

                # Get the reward for this timestep
                # REVISIT: For now, only using the same single kart on the team
                reward = get_reward(player_states[0], player_states[1:], opponent_states, state_data_dictionaries["soccer_state"], agent_team_num)

                # Train the state agent for this timestep
                state_agent.train_immediate_timestep(prev_state, prediction_actions, reward, player_features, False)

                # Save this timestep into state agent memory
                state_agent.memorize(prev_state, prediction_actions, reward, player_features, False)

                # Set previous state for next iteration
                prev_state = player_features

            # The game is done if we've left the load_recording loop and memorized something
            if len(state_agent.memory):

                # Increment state agent games
                state_agent.n_games += 1

                state_agent.train_batch_timesteps()

                # get the final score
                agent_score = get_score(state_data_dictionaries["soccer_state"], agent_team_num)
                opponent_score = get_score(state_data_dictionaries["soccer_state"], opponent_team_num)

                print("Game %0d - StateAgent (Team %0d) scored %0d versus %s (Team %0d) scored %0d" % 
                        (state_agent.n_games, agent_team_num, agent_score, training_opponent, opponent_team_num, opponent_score))

                # Save this model if the score improved (or stayed the same)
                if agent_score >= best_score:
                    best_score = agent_score
                    save_model(state_agent.model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', default=LOGDIR_PATH, help="Log Directory for tensorboard logs. *Currently not being used*")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate of the model")
    parser.add_argument('-g', '--gamma', type=float, default=0.9, help="Discount Rate for Reinforcement learning")
    parser.add_argument('-opt', '--optimizer', type=str, default="ADAM", choices=["ADAM", "SGD"], help="Name of Optimizer to use for training. Supported options: 'ADAM', 'SGD'")
    parser.add_argument('-ep', '--epochs', type=int, default=1, help="Number of epochs to train model over")
    parser.add_argument('-nt', '--n_trajectories', type=int, default=1, help="Number of trajectories to rollout per epoch. Be careful going too high on this to avoid running out of memory!")
    parser.add_argument('-ld', '--load_model', action='store_true', help="Load an existing state_agent model to continue training. Using state_agent/state_agent.pt")
    parser.add_argument('--training_opponent', type=str, default="random", choices=["random", "state_agent"]+TRAINING_OPPONENT_LIST, help="Training opponent for state_agent per epoch. Defaults to random opponent")
    parser.add_argument('--agent_team', type=int, default=0, choices=[0,1,2], help="Team number for State agent per epoch. Defaults to 0 that will randomize team number per epoch")
    # TODO: Any more knobs to add?

    args = parser.parse_args()

    train_reinforce(args)
