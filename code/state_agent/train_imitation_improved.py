"""
TODO
    Train on all data again with the formula below
    Add Dagger
    Try discrete steering
    Try the Justin's basic network

Was able to train a single agent to go straight to the puck. It appears that both
the proportion of moving-to puck-scoring goal is important as well as the number
of epochs you train. Important to remember these setting might only apply to
going to the puck.

Sweet spot
data = data_heading_to_puck * 5 + data_moving + data_team_making_a_goal
data = 115 * 5 + 533 + 20
epochs = 20

10 epochs and the agent didn't learn well (went straight) and 40 epochs the
agent learned to turn right too well.

Even with the settings above the initialization of the weights matters a lot
about how well the agent reaches the puck.

Number frames moving: 533
Number frames heading to puck: 115
Number frames team making a goal: 20
"""
import torch
import torch.utils.tensorboard as tb
import os
import numpy as np
from random import shuffle

from state_agent.utils import accuracy, load_model, save_model, get_categorized_data
from state_agent.action_network import ActionNetwork
from state_agent.player import extract_features

LOGDIR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logdir')
TRAINING_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'categorized_data')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'DEVICE: {DEVICE}')

DEBUG_EN = True
ACCURARY_CLOSE_PERCENT = 0.1

IMITATION_TEAM_NUM = 1
OPPONENT_TEAM_NUM = 2 if IMITATION_TEAM_NUM == 1 else 1

IMITATION_TEAM_KEY = "team%0d_state" % IMITATION_TEAM_NUM
OPPONENT_TEAM_KEY = "team%0d_state" % OPPONENT_TEAM_NUM


# TODO: try two different loss function, categorical loss for steering
def train(args, batch_size=128):
    if args.load_model:
        model = load_model()
    else:
        model = ActionNetwork()
        # model = ActionNetwork(training=True)

    model.to(DEVICE)
    train_logger = None
    if args.logdir is not None:
        train_logger = tb.SummaryWriter(os.path.join(args.logdir, 'train'))

    # MSELoss for Continuous Actions e.g. steering & acceleration
    mse_loss_module = torch.nn.MSELoss()

    # Binary Cross Entropy for binary actions e.g. drifting, braking, fire, nitro, and rescue
    # REVISIT: will threshold with 0.5 and use MSELoss for these for now
    # bce_loss_module = torch.nn.BCEWithLogitsLoss()

    # Create Optimizer
    # optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())

    # # Create LR Scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    # Load Data
    data = get_categorized_data(args.dataset, args.colab)
    data = np.array(data)

    global_step = 0

    # Iterate over epochs
    for epoch in range(args.epochs):
        # shuffle(data)

        model.train()

        # For collecting training accuracies
        train_accuracy_list = []
        frame = 0

        # Iterate over each timestep in a game recording
        # State_data_dictionaries keys includes team1_state, team2_state, actions, soccer_state
        # for state_data_dictionaries in data:
        for iteration in range(0, len(data), batch_size):
            frame += 128
            if frame % 256 == 0:
                print(f'Epoch: {epoch + 1}|{args.epochs}, Frame: {frame}|{len(data)}, Prediction: {prediction[0, :]}')

            batch_ids = torch.randint(0, len(data), (batch_size,), device=DEVICE)
            data_batch = data[batch_ids]

            action_labels_batch = []
            player_features_batch = []

            for state_data_dictionaries in data_batch:

                # Get player & opponent state dictionaries
                player_states = state_data_dictionaries[IMITATION_TEAM_KEY]
                opponent_states = state_data_dictionaries[OPPONENT_TEAM_KEY]
                soccer_state = state_data_dictionaries["soccer_state"]

                # Iterate over all players on Imitation team
                for player_num, player_state in enumerate(player_states):

                    # Get features from dictionaries to feed into network
                    player_features = extract_features(
                        player_state,
                        soccer_state,
                        opponent_states,
                        0
                    )

                    # Get action labels from dictionary
                    action_label_dict = state_data_dictionaries["actions"][player_num * 2]
                    action_labels = convert_action_dictionary_to_tensor(action_label_dict)

                    player_features_batch.append(player_features[None])
                    action_labels_batch.append(action_labels[None])

            player_features_batch = torch.cat(player_features_batch, 0)
            action_labels_batch = torch.cat(action_labels_batch, 0)

            # Use CUDA if available to speed up training
            player_features = player_features_batch.to(DEVICE)
            action_labels = action_labels_batch.to(DEVICE)

            # Zero out gradient for this iteration
            optimizer.zero_grad()

            # Forward pass input through model to get prediction
            prediction = model(player_features)

            # Foward pass prediction and heatmap through loss module to get loss
            loss = mse_loss_module(input=prediction, target=action_labels)
            train_accuracy_list.append(accuracy(prediction, action_labels, ACCURARY_CLOSE_PERCENT))

            # Log loss
            train_logger.add_scalar("loss", loss, global_step=global_step)

            # Compute gradient by calling backward()
            loss.backward()

            # Update parameters for gradient descent by calling optimizer.step()
            optimizer.step()

            # Increment global step
            global_step += 1

        # # Log and Update Learning Rate
        # train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        # scheduler.step(loss)

        train_logger.add_scalar('accuracy', np.mean(train_accuracy_list), global_step=global_step)

        if args.colab:
            torch.save(model.state_dict(), 'state_agent.pt')
        else:
            save_model(model, 'state_agent')


# def convert_action_dictionary_to_tensor(action_dictionary):
#     return torch.as_tensor((action_dictionary.get("acceleration", 0),
#                             action_dictionary.get("steer", 0),
#                             action_dictionary.get("brake", 0),
#                             ), dtype=torch.float32)

def convert_action_dictionary_to_tensor(act):
    if act['steer'] == -1:
        action_tensor = torch.as_tensor((act["acceleration"], 1, 0, 0, act["brake"]), dtype=torch.float32)
    elif act['steer'] == 0:
        action_tensor = torch.as_tensor((act["acceleration"], 0, 1, 0, act["brake"]), dtype=torch.float32)
    elif act['steer'] == 1:
        action_tensor = torch.as_tensor((act["acceleration"], 0, 0, 1, act["brake"]), dtype=torch.float32)
    return action_tensor


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', default=LOGDIR_PATH)
    parser.add_argument('-dp', '--dataset', type=str, help="Path to imitation pkl data", default=TRAINING_PATH)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help="Learning rate of the model")
    parser.add_argument('-ep', '--epochs', type=int, default=2, help="Number of epochs to train model over")
    parser.add_argument('-ld', '--load_model', action='store_true', help="Load an existing state_agent model to continue training. Using frozen_agent/frozen_agent.pt")
    parser.add_argument('--colab', action='store_true')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()