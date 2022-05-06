import numpy as np
from psutil import net_connections
import pystk
import os, subprocess, random
import torch

from glob import glob

from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms.functional as TF
from enum import IntEnum

TRAINING_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'imitation_data')
TRAINING_OPPONENT_LIST = ["jurgen_agent", "geoffrey_agent", "yann_agent", "yoshua_agent"]

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = '../../data'

DEBUG_EN = False

def load_module_file(package, file_path):
    import importlib

    spec = importlib.util.spec_from_file_location(package, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class Team(IntEnum):
    RED = 0
    BLUE = 1


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-dp', '--datapath', default=TRAINING_PATH, help="Datapath directory for generated data")
    parser.add_argument('-nt', '--n_trajectories', type=int, default=1, help="Number of trajectories to rollout")
    parser.add_argument('--record_video', action='store_true', help="Record a .mp4 video for every game")
    parser.add_argument('--clean', action='store_true', help="Clean datapath directory")
    parser.add_argument('--team1', type=str, default="random", choices=["random"]+TRAINING_OPPONENT_LIST, help="Team1 agent. Defaults to random agent")
    parser.add_argument('--team2', type=str, default="random", choices=["random"]+TRAINING_OPPONENT_LIST, help="Team2 agent. Defaults to random agent")
    # TODO: Any more knobs to add?

    args = parser.parse_args()

    generate_imitation_data(args)


def generate_imitation_data(args):

    if args.clean:
        clean_pkl_files(args.datapath)

    for i in range(args.n_trajectories):
        rollout(team1=args.team1, team2=args.team2, output_dir=args.datapath, 
                record_state=True, record_video=args.record_video, iteration=i)

# Clean out all pkl files from a directory
def clean_pkl_files(dir):
    if os.path.exists(dir):
        pkl_file_list = get_pickle_files(dir)
        for pkl_file in pkl_file_list:
            os.remove(pkl_file)


# get list of pickle files from a dataset_path
def get_pickle_files(dataset_path=os.path.abspath(os.path.dirname(__file__))):
    return glob(os.path.join(dataset_path, '*.pkl'))


# Rollout just a single game by calling tournament runner using subprocess
def rollout(team1="random", team2="random", output_dir=TRAINING_PATH, record_state=True, record_video=False, iteration=0):

    # Make output_dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Base run command
    run_cmd = ["python", "-m", "tournament.runner"]

    # Set team1 if random
    if team1 == "random":
        team1 = get_random_opponent()

    # Set team2 if random
    if team2 == "random":
        team2 = get_random_opponent()

    # Add teams to run command
    run_cmd += [team1, team2]

    # Construct the name of this rollout
    rollout_name = "%0d_%s_v_%s" % (iteration, team1, team2)

    # Add saving state pkl file if record_state is True
    if record_state:
        state_output = os.path.join(output_dir, "%s.pkl" % rollout_name)
        run_cmd += ["-s", state_output]

    # Add saving .mp4 video file if record_video is True
    if record_video:
        video_output = os.path.join(output_dir, "%s.mp4" % rollout_name)
        run_cmd += ["-r", video_output]
    
    output = subprocess.check_output(run_cmd)


def get_random_opponent():
    return random.choice(TRAINING_OPPONENT_LIST)


# Load state recordings
def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


def clean_pkl_files_and_rollout_many(num_rollouts, training_opponent="random", agent_team_num=1, output_dir=TRAINING_PATH):
    clean_pkl_files(output_dir)
    rollout_many(num_rollouts, training_opponent, agent_team_num, output_dir)


# Rollout a number of games calling tournament runner -j (i.e. --parallel) using subprocess
def rollout_many(num_rollouts, training_opponent="random", agent_team_num=1, output_dir=TRAINING_PATH):

    # Make output_dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, "reinforce_data.pkl")

    # Base run command
    run_cmd = ["python", "-m", "tournament.runner", "-s", output_dir, "-j", str(num_rollouts)]

    # Set training opponent
    if training_opponent == "random":
        training_opponent = get_random_opponent()

    # Rollout with state_agent on appropriate team
    if agent_team_num == 1:
        run_cmd += ["state_agent", training_opponent]
    else:
        run_cmd += [training_opponent, "state_agent"]
    
    # Invoke the run command
    output = subprocess.check_output(run_cmd)


# Returns accuracy between prediction and labels within pct_close
def accuracy(prediction, labels, pct_close):
  n_items = len(labels)
  n_correct = torch.sum((torch.abs(prediction - labels) < pct_close ))
  acc = (n_correct.item() / n_items)  # scalar
  return acc


def video_grid(team_images, team_state=''):
    from PIL import Image, ImageDraw
    grid = np.array(team_images)
    grid = Image.fromarray(grid)
    grid = grid.resize((grid.width // 2, grid.height // 2))

    draw = ImageDraw.Draw(grid)
    draw.text((20, 20), team1_state, fill=(255, 0, 0))
    draw.text((20, grid.height // 2 + 20), team2_state, fill=(0, 0, 255))
    return grid


def map_image(team_state, soccer_state, resolution=512, extent=65, anti_alias=1):
    BG_COLOR = (0xee, 0xee, 0xec)
    RED_COLOR = (0xa4, 0x00, 0x00)
    BLUE_COLOR = (0x20, 0x4a, 0x87)
    BALL_COLOR = (0x2e, 0x34, 0x36)
    from PIL import Image, ImageDraw
    r = Image.new('RGB', (resolution*anti_alias, resolution*anti_alias), BG_COLOR)

    def _to_coord(x):
        return resolution * anti_alias * (x + extent) / (2 * extent)

    draw = ImageDraw.Draw(r)
    # Let's draw the goal line
    draw.line([(_to_coord(x), _to_coord(y)) for x, _, y in soccer_state['goal_line'][0]], width=5*anti_alias, fill=RED_COLOR)
    draw.line([(_to_coord(x), _to_coord(y)) for x, _, y in soccer_state['goal_line'][1]], width=5*anti_alias, fill=BLUE_COLOR)

    # and the ball
    x, _, y = soccer_state['ball']['location']
    s = soccer_state['ball']['size']
    draw.ellipse((_to_coord(x-s), _to_coord(y-s), _to_coord(x+s), _to_coord(y+s)), width=2*anti_alias, fill=BALL_COLOR)

    # and karts

    teams = []
    
    if len(team_state) > 0:    
        teams.append((BLUE_COLOR, team_state[0]))
    if len(team_state) > 1:    
        teams.append((RED_COLOR, team_state[1]))
    for c, s in teams:
        for k in s:
            x, _, y = k['kart']['location']
            fx, _, fy = k['kart']['front']
            sx, _, sy = k['kart']['size']
            s = (sx+sy) / 2
            draw.ellipse((_to_coord(x - s), _to_coord(y - s), _to_coord(x + s), _to_coord(y + s)), width=5*anti_alias, fill=c)
            draw.line((_to_coord(x), _to_coord(y), _to_coord(x+(fx-x)*2), _to_coord(y+(fy-y)*2)), width=4*anti_alias, fill=0)

    if anti_alias == 1:
        return r
    return r.resize((resolution, resolution), resample=Image.ANTIALIAS)


# Recording functionality
class BaseRecorder:
    def __call__(self, team_state, soccer_state, actions, team_images=None):
        raise NotImplementedError

    def __and__(self, other):
        return MultiRecorder(self, other)

    def __rand__(self, other):
        return MultiRecorder(self, other)


class MultiRecorder(BaseRecorder):

    def __init__(self, *recorders):
        self._r = [r for r in recorders if r]

    def __call__(self, *args, **kwargs):
        for r in self._r:
            r(*args, **kwargs)

    @property
    def states(self) -> list:        
        for r in self._r:                        
            if hasattr(r, "states"):
                return r.states
        return []
class VideoRecorder(BaseRecorder):
    """
        Produces pretty output videos
    """
    def __init__(self, video_file):
        import imageio
        self._writer = imageio.get_writer(video_file, fps=20)

    def __call__(self, team_state, soccer_state, actions, team_images=None):
        if team_images:
            self._writer.append_data(np.array(video_grid(team_images,
                                                         'Blue: %d' % soccer_state['score'][1],
                                                         'Red: %d' % soccer_state['score'][0])))
        else:
            self._writer.append_data(np.array(map_image(team_state, soccer_state)))

    def __del__(self):
        if hasattr(self, '_writer'):
            self._writer.close()


class DataRecorder(BaseRecorder):
    def __init__(self, record_images=False):
        self._record_images = record_images
        self._data = []

    def __call__(self, team_state, soccer_state, actions, team_images=None):
        data = dict(team_state=team_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            data['team_images'] = team_images            
        self._data.append(data)

    def data(self):
        return self._data

    def reset(self):
        self._data = []


class StateRecorder(BaseRecorder):
    def __init__(self, state_action_file=None, record_images=False):
        self._record_images = record_images
        if state_action_file:
            self._f = open(state_action_file, 'wb')
        self.states = []

    def __call__(self, team_state, soccer_state, actions, team_images=None):
        from pickle import dump
        data = dict(team_state=team_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            data['team_images'] = team_images         
        if hasattr(self, '_f'):            
            dump(dict(data), self._f)
            self._f.flush()
        self.states.append(data)
        
    def __del__(self):
        if hasattr(self, '_f'):
            self._f.close()

    def get_states(self):
        return self.states

if __name__ == "__main__":
    main()

