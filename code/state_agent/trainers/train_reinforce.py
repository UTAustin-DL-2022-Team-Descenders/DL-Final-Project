import subprocess
import torch
import torch.utils.tensorboard as tb
import os, random
from ..utils import *
from ..agents.basic.state_agent import *

LOGDIR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logs')
TRAINING_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'reinforce_data')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEBUG_EN = False

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', default=LOGDIR_PATH, help="Log Directory for tensorboard logs. *Currently not being used*")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate of the model")
    parser.add_argument('-g', '--discount_rate', type=float, default=0.99, help="Discount Rate for Reinforcement learning")
    parser.add_argument('-opt', '--optimizer', type=str, default="ADAM", choices=["ADAM", "SGD"], help="Name of Optimizer to use for training. Supported options: 'ADAM', 'SGD'")
    parser.add_argument('-ep', '--epochs', type=int, default=1, help="Number of epochs to train model over")
    parser.add_argument('-nt', '--n_trajectories', type=int, default=1, help="Number of trajectories to rollout per epoch. Be careful going too high on this to avoid running out of memory!")
    parser.add_argument('-ld', '--load_model', action='store_true', help="Load an existing state_agent model to continue training. Using state_agent/state_agent.pt")
    parser.add_argument('-to', '--training_opponent', type=str, default="random", choices=["random", "state_agent"]+TRAINING_OPPONENT_LIST, help="Training opponent for state_agent per epoch. Defaults to random opponent")
    parser.add_argument('-at', '--agent_team', type=int, default=0, choices=[0,1,2], help="Team number for State agent per epoch. Defaults to 0 that will randomize team number per epoch")
    parser.add_argument('-rv', '--record_video_cadence', type=int, default=10, help="Number of games between recording video while training")
    parser.add_argument('-std', '--noise_std', type=int, default=0.01, help="Standard Deviation for normalized exploration noise")

    args = parser.parse_args()

    train_reinforce(args)


def train_reinforce(args):

    # Base case: negative goals scored (should always do better)
    best_score = -1
    best_reward = float("-inf")

    train_logger = None
    if args.logdir is not None:
        train_logger = tb.SummaryWriter(os.path.join(args.logdir, "reinforce"), flush_secs=1)

    # Create StateAgent
    state_agent = StateAgent(load_existing_model=args.load_model, 
                        optimizer=args.optimizer, 
                        lr=args.learning_rate,
                        discount_rate=args.discount_rate,
                        logger=train_logger,
                        player_num=0,
                        noise_std=args.noise_std
                        )
   

    global_step = 0

    for _ in range(args.epochs):
    #while True:

        # Set training opponent so it's visible to end of game results print
        training_opponent = args.training_opponent if args.training_opponent != "random" else get_random_opponent()

        # Set Agent team number. Randomized if args.agent_team == 0
        agent_team_num = args.agent_team if args.agent_team != 0 else random.choice([1,2])
        opponent_team_num = 2 if agent_team_num == 1 else 1

        # Clean out old pkl files and rollout a number of games 
        # and save recordings into pickle files
        clean_pkl_files_and_rollout_many(args.n_trajectories, 
                                        training_opponent=training_opponent, 
                                        agent_team_num=agent_team_num,
                                        output_dir=TRAINING_PATH)

        # Periodically record video to see how the agent is doing
        if state_agent.n_games > 0 and state_agent.n_games % args.record_video_cadence == 0:
            rollout_evaluation(agent_team_num, training_opponent, state_agent.n_games)

            
        training_pkl_file_list = get_pickle_files(TRAINING_PATH)

        # Iterate over game trajectories
        for pkl_file in training_pkl_file_list:

            # Initialize curr_state_dictionaries and next_state_dictionaries. 
            # These will be used as fast & slow pointers to timestep state dictionaries
            curr_state_dictionaries = next_state_dictionaries = None

            # Initialize game reward
            game_reward = 0

            if DEBUG_EN:
                print("train_reinforce - pkl_file: %s" % pkl_file)

            # Iterate over each timestep in a game recording
            # State_data_dictionaries keys includes team1_state, team2_state, actions, soccer_state
            for state_data_dictionaries in load_recording(pkl_file):

                # For first timestep when curr_state_dictionaries is None, 
                # save state data dictionary and continue
                if curr_state_dictionaries == None:
                    curr_state_dictionaries = state_data_dictionaries
                    continue
        
                # given that curr_state_dictionaries is lagging by a timestep,
                # next_state_dictionaries will be represented by this iteration's state_data_dictionaries
                next_state_dictionaries = state_data_dictionaries
                next_state_features = get_features_from_unified_state_dictionaries(next_state_dictionaries, agent_team_num, state_agent.player_num)

                # Iterate over players on a team
                #for player_num in range(len(player_states)):
                # REVIST: Just using player_num set to 0 for now

                # Get features from dictionaries to feed into network
                curr_state_features = get_features_from_unified_state_dictionaries(curr_state_dictionaries, agent_team_num, state_agent.player_num)

                # Get action tensor for current state
                curr_state_actions = state_agent.get_action_tensor(curr_state_features)

                # Get the reward for current state timestep
                reward = get_reward(next_state_dictionaries, curr_state_actions, agent_team_num, state_agent.player_num) 

                # Add this reward to this game's reward total
                game_reward += reward

                # Save this timestep into state agent memory
                state_agent.memorize(curr_state_features, curr_state_actions, reward, next_state_features, not_done=True)

                # Train the state agent over a batch of timesteps
                # To apply BatchNorm1d in networks, need memory to be > 1
                if len(state_agent.memory) > 1:
                    state_agent.train_batch_timesteps(global_step)

                # Set current state for next iteration
                curr_state_dictionaries = next_state_dictionaries
                global_step += 1

            # The game is not_done if the agent memorized something
            # and curr_state_dictionaries and next_state_dictionaries match
            if len(state_agent.memory) > 0 and curr_state_dictionaries == next_state_dictionaries:

                curr_state_features = next_state_features = get_features_from_unified_state_dictionaries(curr_state_dictionaries, agent_team_num, state_agent.player_num)

                # Get action tensor for current state
                curr_state_actions = state_agent.get_action_tensor(curr_state_features)

                # Get the reward for current state timestep
                reward = get_reward(next_state_dictionaries, curr_state_actions, agent_team_num, state_agent.player_num) 

                # Add this reward to this game's reward total
                game_reward += reward

                # Memorize game ending timestep
                state_agent.memorize(curr_state_features, curr_state_actions, reward, next_state_features, not_done=False)

                # Train over a batch of timesteps at the end of the game
                state_agent.train_batch_timesteps(global_step)

                # Increment state agent games
                state_agent.n_games += 1

                # get the final score
                agent_score = get_score(state_data_dictionaries["soccer_state"], agent_team_num)
                opponent_score = get_score(state_data_dictionaries["soccer_state"], opponent_team_num)

                print("Game %0d - StateAgent (Team %0d) scored %0d versus %s (Team %0d) scored %0d reward %0d" % 
                        (state_agent.n_games, agent_team_num, agent_score, training_opponent, opponent_team_num, opponent_score, game_reward))

                train_logger.add_scalar("total reward", game_reward, global_step=global_step)

                # Save this model if the reward improved
                if game_reward > best_reward:
                    best_reward = game_reward
                    save_model(state_agent.action_net)
                    save_model(state_agent.critic_net, "critic")
                    print("best_reward improved to %0d. Saving state_agent & critic" % best_reward)
                # Save this model if the score improved (or stayed the same)
                #if agent_score >= best_score:
                #    best_score = agent_score
                #    save_model(state_agent.action_net)
                #    if agent_score == 3:
                #        print("Wow the agent scored 3 goals! Take the dub and get out of the casino")
                #        return

def rollout_evaluation(agent_team_num, training_opponent, n_game):

    if agent_team_num == 1:
        team1 = "state_agent"
        team2 = training_opponent
    else:
        team2 = "state_agent"
        team1 = training_opponent

    rollout(team1, team2, os.path.abspath(os.path.dirname(__file__)), record_state=False, record_video=True, iteration=n_game)
    print("Game %0d recorded" % n_game)



if __name__ == '__main__':
    main()
