from .. import ActionNetwork, save_model, load_model
from ..state_agent import *
import torch, os, sys
import torch.utils.tensorboard as tb
import numpy as np
from ..utils import accuracy, get_pickle_files, load_recording

LOGDIR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logs')
TRAINING_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'imitation_data')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DEBUG_EN = False
ACCURARY_CLOSE_PERCENT = 0.1

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', default=LOGDIR_PATH)
    parser.add_argument('-dp', '--dataset', type=str, help="Path to imitation pkl data", default=TRAINING_PATH)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help="Learning rate of the model")
    parser.add_argument('-ep', '--epochs', type=int, default=2, help="Number of epochs to train model over")
    parser.add_argument('-ld', '--load_model', action='store_true', help="Load an existing state_agent model to continue training. Using state_agent/state_agent.pt")
    parser.add_argument('--imitation_team', type=int, choices=[1,2], default=1, help="Team the state_agent will attempt to imitate")

    args = parser.parse_args()

    train(args)

def train(args):

    if args.load_model:
        model = load_model()
    else:
        model = ActionNetwork()

    model.to(DEVICE)
    train_logger = None
    if args.logdir is not None:
        train_logger = tb.SummaryWriter(os.path.join(args.logdir, "imitation"), flush_secs=1)

    # Get model parameters
    parameters = model.parameters()

    # MSELoss for Continuous Actions e.g. steering & acceleration
    mse_loss_module = torch.nn.MSELoss()

    # Binary Cross Entropy for binary actions e.g. drifting, braking, fire, nitro, and rescue
    bce_loss_module = torch.nn.BCEWithLogitsLoss()

    # Create Optimizer
    #optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(parameters)
    
    # Create LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    # Load Data
    training_pkl_file_list = get_pickle_files(args.dataset)

    if not training_pkl_file_list:
        print("No .pkl files found in %s" % args.dataset)
        sys.exit(-1)

    global_step = 0

    # Iterate over epochs
    for epoch in range(args.epochs):

        model.train()

        # For collecting training accuracies
        train_accuracy_list = []

        # Set opponent team number for indexing into state dictionaries
        opponent_team = 2 if args.imitation_team == 1 else 1

        # Get team keys for accessing state dictionaries
        imitation_team_key = "team%0d_state" % args.imitation_team
        opponent_team_key = "team%0d_state" % opponent_team

        # Iterate over each game
        for pkl_file in training_pkl_file_list:

            if DEBUG_EN:
                print("pkl_file - ", pkl_file)

            # Iterate over each timestep in a game recording
            # State_data_dictionaries keys includes team1_state, team2_state, actions, soccer_state
            for state_data_dictionaries in load_recording(pkl_file):

                # Get player & opponent state dictionaries
                player_states = state_data_dictionaries[imitation_team_key]
                opponent_states = state_data_dictionaries[opponent_team_key]

                # Iterate over all players on Imitation team
                # REVISIT: could we differentiate which sub network (offense/defense) 
                # is trained by doing something like even player_num are sent to 
                # offense network & odd player_num to defense network?
                for player_num, player_state in enumerate(player_states):
                
                    # Get features from dictionaries to feed into network
                    player_features = get_features(player_state, player_states[:player_num] + player_states[player_num+1:], 
                                                    opponent_states, state_data_dictionaries["soccer_state"],
                                                    args.imitation_team)

                    if args.imitation_team == 1:
                        player_action_index = 0 if player_num == 0 else 2
                    else:
                        player_action_index = 1 if player_num == 0 else 3

                    # Get action labels from dictionary
                    action_label_dict = state_data_dictionaries["actions"][player_action_index]
                    action_labels = convert_action_dictionary_to_tensor(action_label_dict)
                
                    # Use CUDA if available to speed up training
                    player_features = player_features.to(DEVICE)
                    action_labels = action_labels.to(DEVICE)

                    # Zero out gradient for this iteration
                    optimizer.zero_grad()

                    # Forward pass input through model to get prediction
                    prediction = model(player_features)[0]

                    if DEBUG_EN:
                        print("prediction    - ", prediction)
                        print("action_labels - ", action_labels)

                    # Foward pass prediction actions and labels through loss module to get loss

                    # Continuous Actions (Acceleration & Steering) use MSE
                    acc_loss = mse_loss_module(input=prediction[0], target=action_labels[0])
                    steer_loss = mse_loss_module(input=prediction[1], target=action_labels[1])

                    # Binary Actions (Brake, Drift, Fire, Nitro) use BCE
                    brake_loss = bce_loss_module(input=prediction[2], target=action_labels[2])
                    drift_loss = bce_loss_module(input=prediction[3], target=action_labels[3])
                    fire_loss = bce_loss_module(input=prediction[4], target=action_labels[4])
                    nitro_loss = bce_loss_module(input=prediction[5], target=action_labels[5])

                    loss = acc_loss + steer_loss + brake_loss + drift_loss + fire_loss + nitro_loss

                    train_accuracy_list.append(accuracy(prediction, action_labels, ACCURARY_CLOSE_PERCENT))

                    # Log loss
                    train_logger.add_scalar("loss", loss, global_step=global_step)

                    # Compute gradient by calling backward()
                    loss.backward()

                    # Update parameters for gradient descent by calling optimizer.step()
                    optimizer.step()

                    # Increment global step
                    global_step += 1

        # Log and Update Learning Rate
        train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        scheduler.step(loss)

        train_logger.add_scalar('accuracy', np.mean(train_accuracy_list), global_step=global_step)

    save_model(model)


# Assumes network outputs 6 different actions
# Output channel order: [acceleration, steer, brake, drift, fire, nitro]
# Uses default value of 0 if key is not found
def convert_action_dictionary_to_tensor(action_dictionary):
    action_tensor = torch.as_tensor((action_dictionary.get("acceleration", 0),
                            action_dictionary.get("steer", 0),
                            action_dictionary.get("brake", 0),
                            action_dictionary.get("drift", 0),
                            action_dictionary.get("fire", 0),
                            action_dictionary.get("nitro", 0)),
                        dtype=torch.float32, device=DEVICE)
    
    return torch.unsqueeze(action_tensor, 0)

if __name__ == '__main__':
    main()
