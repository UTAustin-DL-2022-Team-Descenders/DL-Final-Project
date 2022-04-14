from utils.track import state_features, three_points_on_track,cart_direction, cart_lateral_distance
import torch
import sys, os
import pystk
import ray
import numpy as np
from PIL import Image, ImageFont, ImageDraw

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
font = ImageFont.load_default()

@ray.remote
class Rollout:
    def __init__(self, screen_width, screen_height, hd=True, track='lighthouse', render=True, frame_skip=1):
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
        race_config = pystk.RaceConfig(track=track)
        self.race = pystk.Race(race_config)
        self.race.start()
    
    def __call__(self, agent, n_steps=200):
        torch.set_num_threads(1)
        self.race.restart()
        self.race.step()
        data = []
        track_info = pystk.Track()
        track_info.update()

        for i in range(n_steps // self.frame_skip):
            world_info = pystk.WorldState()
            world_info.update()

            # Gather world information
            kart_info = world_info.players[0].kart

            agent_data = {'track_info': track_info, 'kart_info': kart_info}
            if self.render:
                agent_data['image'] = np.array(self.race.render_data[0].image)

            # Act
            action = agent(**agent_data)
            agent_data['action'] = action

            # Take a step in the simulation
            for it in range(self.frame_skip):
                self.race.step(action)

            # Save all the relevant data
            data.append(agent_data)
        return data

def show_video(data, fps=30):
    import imageio
    from IPython.display import Video, display 
    
    frames = [d['image'] for d in data]    
    distances = [t['kart_info'].distance_down_track for t in data]
    features = [state_features(**t) for t in data]
    directions = [cart_direction(t['kart_info']) for t in data]
    laterals = [cart_lateral_distance(t['kart_info'], three_points_on_track(t['kart_info'].distance_down_track, t['track_info'])) for t in data]

    images = []
    for frame, feature, distance, direction, lateral in zip(frames, features, distances, directions, laterals):
        img = Image.fromarray(frame)        
        image_to_edit = ImageDraw.Draw(img)
        
        image_to_edit.text((10, 10), "Distance: {}".format(distance))
        image_to_edit.text((10, 20), "direction: {}".format(direction))         
        image_to_edit.text((10, 30), "lateral: {}".format(lateral))         
        images.append(np.array(img))

    imageio.mimwrite('/tmp/test.mp4', images, fps=fps, bitrate=1000000)
    display(Video('/tmp/test.mp4', width=800, height=600, embed=True))

viz_rollout = Rollout.remote(400, 300)
def show_agent(agent, n_steps=600):
    data = ray.get(viz_rollout.__call__.remote(agent, n_steps=n_steps))
    show_video(data)
    
rollouts = [Rollout.remote(50, 50, hd=False, render=False, frame_skip=5) for i in range(10)]
def rollout_many(many_agents, **kwargs):
    ray_data = []
    for i, agent in enumerate(many_agents):
         ray_data.append( rollouts[i % len(rollouts)].__call__.remote(agent, **kwargs) )
    return ray.get(ray_data)

def dummy_agent(**kwargs):
    action = pystk.Action()
    action.acceleration = 1
    return action
