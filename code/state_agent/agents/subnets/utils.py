# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

from .features import SoccerFeatures, cart_location, get_puck_center, cart_speed
import torch
import sys, os
import pystk
import ray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from state_agent.utils import map_image
from state_agent.runner import to_native

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
font = ImageFont.load_default()
    

# tried to break this into multiple classes, but Ray doesn't seem to work when there's a subclass used
@ray.remote
class Rollout:
    def __init__(self, screen_width, screen_height, hd=True, track='lighthouse', render=True, frame_skip=1, 
                 mode="track"):
        # Init supertuxkart
        if not render:
            config = pystk.GraphicsConfig.none()
        elif hd:
            config = pystk.GraphicsConfig.hd()
        else:
            config = pystk.GraphicsConfig.ld()
        config.screen_width = screen_width
        config.screen_height = screen_height
        pystk.init(config)
        
        self.frame_skip = frame_skip
        self.render = render
        self.track_info = None  
        self.mode = mode   
        
        self.create_race(track)
    
    def create_race(self, track):
        race_config = None
        if self.mode == "track":
            race_config = pystk.RaceConfig(track=track)
        elif self.mode == "soccer":
            race_config = pystk.RaceConfig(track="icy_soccer_field", mode=pystk.RaceConfig.RaceMode.SOCCER)
            race_config.players.pop()
            race_config.players.append(self._make_config(0, False, "tux"))
        self.race = pystk.Race(race_config)
        self.race.start()  

    def initialize_state(self, world_info, randomize=False, ball_location=None, ball_velocity=None, player_location=None, **kwargs):
        if self.mode == "track":
            self.track_info = track_info = pystk.Track()
            track_info.update()
        elif self.mode == "soccer":
            ball_location = ball_location if ball_location else ([0, 0] if randomize == False else np.random.normal(loc=0.0, scale=24.0, size=(2)))
            ball_velocity = ball_velocity if ball_velocity else [0, 0]
            world_info.set_ball_location((ball_location[0], 1, ball_location[1]),
                                        (ball_velocity[0], 0, ball_velocity[1]))
            if player_location:
                world_info.set_kart_location(0, player_location, [0, 0, 0, 1.0], 0)

    def agent_data(self, world_info):
        # Gather world information
        kart_info = world_info.players[0].kart

        agent_data = {}    
        if self.mode == "track":
            agent_data = {'track_info': self.track_info, 'kart_info': kart_info}
        elif self.mode == "soccer":
            agent_data = {'track_info': None, 'soccer_state': world_info.soccer, 'kart_info': kart_info}

        if self.render:
            agent_data['image'] = np.array(self.race.render_data[0].image)
            if self.mode == "soccer":
                agent_data['map'] = map_image([[to_native(world_info.players[0])]], to_native(world_info.soccer))

        return agent_data

    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    def __call__(self, agent, n_steps=600, **kwargs):
        torch.set_num_threads(1)
        self.race.restart()
        self.race.step()
        data = []

        world_info = pystk.WorldState()    
        self.initialize_state(world_info, **kwargs)
        world_info.update()
    
        for i in range(n_steps // self.frame_skip):            
            world_info = pystk.WorldState()        
            world_info.update()

            agent_data = self.agent_data(world_info)

            # Act
            action = agent(**agent_data)

            game_action = pystk.Action(
                **dict(vars(action))
            )

            agent_data['action'] = action

            # Take a step in the simulation
            for it in range(self.frame_skip):
                self.race.step(game_action)

            # Save all the relevant data
            data.append(agent_data)
        return data

soccer_feature_extractor = SoccerFeatures()

def show_video_soccer(data, fps=30):
    import imageio
    from IPython.display import Video, display 
    
    frames = [d['image'] for d in data]    
    frames_map = [d['map'] for d in data]    
    actions = [t['action'] for t in data]
    features = [soccer_feature_extractor.get_feature_vector(**t) for t in data]
    distances = [np.linalg.norm(get_puck_center(t['soccer_state']) - cart_location(t['kart_info'])) for t in data]
    speeds = [cart_speed(t['kart_info']) for t in data]

    images = []
    map_images = []
    for frame, action, distance, feature, speed in zip(frames, actions, distances, features, speeds):
        img = Image.fromarray(frame)        
        image_to_edit = ImageDraw.Draw(img)        
        image_to_edit.text((10, 10), "accel: {}".format(float(action.acceleration)))         
        image_to_edit.text((10, 20), "speed: {}".format(speed))         
        image_to_edit.text((10, 30), "steering: {}".format(float(action.steer)))         
        image_to_edit.text((10, 40), "drift: {}".format(action.drift))         
        image_to_edit.text((10, 50), "distance: {}".format(distance))         
        image_to_edit.text((10, 60), "angle diff: {}".format(soccer_feature_extractor.select_delta_steering(feature)))         
        images.append(np.array(img))

    for img, action, distance, feature, speed in zip(frames_map, actions, distances, features, speeds):
        image_to_edit = ImageDraw.Draw(img)    
        image_to_edit.text((10, 10), "accel: {}".format(float(action.acceleration)), fill=(0, 0, 0))
        image_to_edit.text((10, 20), "speed: {}".format(speed), fill=(0, 0, 0))         
        image_to_edit.text((10, 30), "steering: {}".format(float(action.steer)), fill=(0, 0, 0))
        image_to_edit.text((10, 40), "drift: {}".format(action.drift), fill=(0, 0, 0))  
        image_to_edit.text((10, 50), "distance: {}".format(distance), fill=(0, 0, 0))    
        image_to_edit.text((10, 60), "angle diff: {}".format(soccer_feature_extractor.select_delta_steering(feature)), fill=(0, 0, 0))                     
        map_images.append(np.array(img))

    # create map video

    imageio.mimwrite('/tmp/test.mp4', images, fps=fps, bitrate=1000000)
    imageio.mimwrite('/tmp/test2.mp4', map_images, fps=fps, bitrate=1000000)
    display(Video('/tmp/test.mp4', width=800, height=600, embed=True))
    display(Video('/tmp/test2.mp4', width=800, height=600, embed=True))


def show_graph(data):

    steer = [t['action'].steer for t in data]
    drift = [t['action'].drift for t in data]
    accel = [t['action'].acceleration for t in data]
    fig, (steering_p, drift_p, accel_p) = plt.subplots(1, 3)
    steering_p.plot(steer)
    steering_p.set_title("Steering")
    drift_p.plot(drift)
    drift_p.set_title("Drift")
    accel_p.plot(accel)
    accel_p.set_title("Accel")
    fig.show()

def show_trajectory_histogram(trajectories, metric, min=0, max=1000, bins=10):
    # histogram of trajectory overall scoring distribution
    scores = [metric(t[-1]["kart_info"]) for t in trajectories]
    scores = np.clip(scores, min, max)
    plt.hist(scores, range(min, max, (max - min) // bins), density=True)
    plt.show()

def show_steering_graph(data):
    import matplotlib.pyplot as plt

    steer = [t['action'].steer for t in data]
    plt.plot(steer)
    plt.show()

viz_rollout_soccer = Rollout.remote(400, 300, mode="soccer")
def run_soccer_agent(agent, rollout=viz_rollout_soccer, **kwargs):
    data = ray.get(rollout.__call__.remote(agent, **kwargs))
    show_video_soccer(data)
    show_graph(data)
    return data

viz_rollouts = [Rollout.remote(50, 50, hd=False, render=False, frame_skip=5, mode="soccer") for i in range(4)]
def rollout_many(many_agents, **kwargs):    
    ray_data = []
    for i, agent in enumerate(many_agents):
         ray_data.append(viz_rollouts[i % len(viz_rollouts)].__call__.remote(agent, **kwargs) )    
    return ray.get(ray_data)

def dummy_agent(**kwargs):
    action = pystk.Action()
    action.acceleration = 1
    return action
        

# StateAgent agnostic Save & Load model functions. Used in state_agent.py Match
def save_model(model, f_path, file=__file__):
    from os import path
    model_scripted = torch.jit.script(model)
    model_scripted.save(path.join(path.dirname(path.abspath(file)), f_path ))


def load_model(f_path, file=__file__):
    from os import path
    import sys
    load_path = path.join(path.dirname(path.abspath(file)), f_path)
    try:
        model = torch.jit.load(load_path)
        #print("Loaded pre-existing ActionNetwork from", load_path)
        return model
    except FileNotFoundError as e:
        return None
    except ValueError as e:
        #print("Couldn't find existing model in %s" % load_path)
        return None

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)