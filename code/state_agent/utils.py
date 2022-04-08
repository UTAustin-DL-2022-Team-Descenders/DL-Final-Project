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
DATASET_PATH = '../../data'

DEBUG_EN = False


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



# TODO: Is this compatiable with load_recording use of yield?
class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH):
        self.data = []
        for state_pkl_file in glob(os.path.join(dataset_path, '*.pkl')):
            self.data.append(state_pkl_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state_pkl_file = self.data[idx]
        yield load_recording(state_pkl_file)


# TODO: Is SuperTuxDataset and by extension this function compatible with yield from load_recordings
def load_data(dataset_path=DATASET_PATH, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

# get list of pickle files from a dataset_path
def get_pickle_files(dataset_path=DATASET_PATH):
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

# TODO: Do we need this?
class PyTux:
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def rollout(self, track, controller, planner=None, max_frames=1000, verbose=False, data_callback=None):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

        for t in range(max_frames):

            state.update()
            track.update()

            kart = state.players[0].kart

            if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
                if verbose:
                    print("Finished at t=%d" % t)
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, track)
            aim_point_image = self._to_image(aim_point_world, proj, view)
            if data_callback is not None:
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            if planner:
                image = np.array(self.k.render_data[0].image)
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            current_vel = np.linalg.norm(kart.velocity)
            action = controller(aim_point_image, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                if planner:
                    ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                    ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
                plt.pause(1e-3)

            self.k.step(action)
            t += 1
        return t, kart.overall_distance / track.length

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()

