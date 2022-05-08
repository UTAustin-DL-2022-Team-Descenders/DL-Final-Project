import torch
from glob import glob
import os
import time


def save_model(model, file):
    from os import path
    model_scripted = torch.jit.script(model)
    model_scripted.save(path.join(path.dirname(path.abspath(__file__)), file + '.pt'))


def load_model():
    from os import path
    load_path = path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt')
    try:
        model = torch.jit.load(load_path)
        print("Loaded pre-existing ActionNetwork from", load_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError("Couldn't find existing model in %s" % load_path)
    except ValueError:
        raise ValueError("Couldn't find existing model in %s" % load_path)


def get_pickle_files(dataset_path=os.path.abspath(os.path.dirname(__file__))):
    files = glob(os.path.join(dataset_path, '*.pkl'))
    files.sort()
    return files


def get_categorized_data(dataset_path, colab=False):
    """
    Time to load: ~6.4 min
    frames_heading_to_puck.pkl load time: 46.5
    frames_moving.pkl load time: 374.5
    frames_team_making_a_goal.pkl load time: 06.8
    """
    if colab:
        import pickle5 as pickle
    else:
        import pickle

    data_moving = get_data_moving(dataset_path, pickle, file='frames_moving.pkl')
    data_heading_to_puck = get_data_heading_to_puck(dataset_path, pickle)
    data_team_making_a_goal = get_data_team_making_a_goal(dataset_path, pickle)

    factor = len(data_moving) // len(data_heading_to_puck)
    print(f'Factor: {factor}')

    data = data_heading_to_puck * factor + data_moving + data_team_making_a_goal
    # data = data_team_making_a_goal
    return data


def get_data_moving(dataset_path, pickle, file='frames_moving.pkl'):
    now = time.time()
    with open(os.path.join(dataset_path, file), 'rb') as f:
        data_moving = pickle.load(f)
    print(f'{file} load time: {time.time() - now:04.1f}')
    print('Number frames moving:', len(data_moving))
    return data_moving


def get_data_heading_to_puck(dataset_path, pickle):
    now = time.time()
    with open(os.path.join(dataset_path, 'frames_heading_to_puck.pkl'), 'rb') as f:
        data_heading_to_puck = pickle.load(f)
    print(f'frames_heading_to_puck.pkl load time: {time.time() - now:04.1f}')
    print('Number frames heading to puck:', len(data_heading_to_puck))
    return data_heading_to_puck


def get_data_team_making_a_goal(dataset_path, pickle):
    now = time.time()
    with open(os.path.join(dataset_path, 'frames_team_making_a_goal.pkl'), 'rb') as f:
        data_team_making_a_goal = pickle.load(f)
    print(f'frames_team_making_a_goal.pkl load time: {time.time() - now:04.1f}')
    print('Number frames team making a goal:', len(data_team_making_a_goal))
    return data_team_making_a_goal


# Returns accuracy between prediction and labels within pct_close
def accuracy(prediction, labels, pct_close):
    n_items = len(labels)
    n_correct = torch.sum((torch.abs(prediction - labels) < torch.abs(pct_close * labels)))
    acc = (n_correct.item() * 100.0 / n_items)  # scalar
    return acc


def check_reached_puck(
        soccer_state=None,
        player_states=None,
        y=None, x0=None, x1=None
):
    if soccer_state is not None and player_states is not None:
        y = soccer_state['ball']['location']
        x0 = player_states[0]['kart']['location']
        x1 = player_states[1]['kart']['location']
    else:
        assert all([a is not None for a in [y, x0, x1]])

    dist0 = ((x0[0] - y[0]) ** 2 + (x0[2] - y[2]) ** 2) ** (1 / 2)
    dist1 = ((x1[0] - y[0]) ** 2 + (x1[2] - y[2]) ** 2) ** (1 / 2)
    reached_puck = True if dist0 < 3 or dist1 < 3 else False
    # print(player_states[0]['kart']['location'][0], player_states[0]['kart']['location'][2], dist0)
    return reached_puck