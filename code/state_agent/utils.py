import numpy as np
import pystk
import os
import torch

from glob import glob

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from enum import IntEnum


RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15

class Team(IntEnum):
    RED = 0
    BLUE = 1


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
    def __init__(self, state_action_file, record_images=False):
        self._record_images = record_images
        self._f = open(state_action_file, 'wb')

    def __call__(self, team_state, soccer_state, actions, team_images=None):
        from pickle import dump
        data = dict(team_state=team_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            data['team_images'] = team_images            
        dump(dict(data), self._f)
        self._f.flush()

    def __del__(self):
        if hasattr(self, '_f'):
            self._f.close()


# get list of pickle files from a dataset_path
def get_pickle_files(dataset_path=os.path.abspath(os.path.dirname(__file__))):
    return glob(os.path.join(dataset_path, '*.pkl'))


# Load state recordings
def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


# Returns accuracy between prediction and labels within pct_close
def accuracy(prediction, labels, pct_close):
  n_items = len(labels)
  n_correct = torch.sum((torch.abs(prediction - labels) < torch.abs(pct_close * labels)))
  acc = (n_correct.item() * 100.0 / n_items)  # scalar
  return acc
