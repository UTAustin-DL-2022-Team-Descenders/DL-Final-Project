from utils.rewards import SoccerBallDistanceObjective
from utils.actors import Agent, TrainingAgent
from utils.rewards import ObjectiveEvaluator, OverallDistanceObjective, TargetDistanceObjective
from utils.track import state_features, state_features_soccer, three_points_on_track,cart_direction, \
    cart_lateral_distance, get_obj1_to_obj2_angle, cart_location, cart_angle, get_puck_center, \
    cart_overall_distance
import torch
import sys, os
import pystk
import ray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from lib.state_agent.utils import map_image
from lib.state_agent.runner import to_native

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
font = ImageFont.load_default()
    

# tried to break this into multiple classes, but Ray doesn't seem to work when there's a subclass used
@ray.remote
class Rollout:
    def __init__(self, screen_width, screen_height, hd=True, track='lighthouse', render=True, frame_skip=1, 
                 mode="track",
                 ball_location=None, ball_velocity=None):
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
        self.ball_location=ball_location
        self.ball_velocity=ball_velocity   

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

    def initialize_state(self, world_info, randomize=False):
        if self.mode == "track":
            self.track_info = track_info = pystk.Track()
            track_info.update()
        elif self.mode == "soccer":
            ball_location = self.ball_location if self.ball_location else ([0, 0] if randomize == False else np.random.normal(loc=0.0, scale=24.0, size=(2)))
            ball_velocity = self.ball_velocity if self.ball_velocity else [0, 0]
            world_info.set_ball_location((ball_location[0], 1, ball_location[1]),
                                        (ball_velocity[0], 0, ball_velocity[1]))

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

    def __call__(self, agent, n_steps=600, randomize=False, **kwargs):
        torch.set_num_threads(1)
        self.race.restart()
        self.race.step()
        data = []

        world_info = pystk.WorldState()    
        self.initialize_state(world_info, randomize=randomize)
        world_info.update()
    
        for i in range(n_steps // self.frame_skip):            
            world_info = pystk.WorldState()        
            world_info.update()

            agent_data = self.agent_data(world_info)

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
    distances = [t['kart_info'].overall_distance for t in data]
    actions = [t['action'] for t in data]
    features = [state_features(**t) for t in data]
    directions = [cart_direction(t['kart_info']) for t in data]
    laterals = [cart_lateral_distance(t['kart_info'], three_points_on_track(t['kart_info'].distance_down_track, t['track_info'])) for t in data]
    angles = [(cart_angle(t['kart_info']), get_obj1_to_obj2_angle(cart_location(t['kart_info']), three_points_on_track(t['kart_info'].distance_down_track, t['track_info'])[1])) for t in data]
    actions = [t['action'] for t in data]

    images = []
    for frame, feature, distance, direction, lateral, action, angle in zip(frames, features, distances, directions, laterals, actions, angles):
        img = Image.fromarray(frame)        
        image_to_edit = ImageDraw.Draw(img)
        
        image_to_edit.text((10, 10), "Distance: {}".format(distance))
        image_to_edit.text((10, 20), "angle: {}, target angle: {}".format(angle[0], angle[1]))
        image_to_edit.text((10, 30), "lateral: {}".format(lateral))         
        image_to_edit.text((10, 40), "steering: {}".format(action.steer))         
        image_to_edit.text((10, 50), "drift: {}".format(action.drift))         
        images.append(np.array(img))

    imageio.mimwrite('/tmp/test.mp4', images, fps=fps, bitrate=1000000)
    display(Video('/tmp/test.mp4', width=800, height=600, embed=True))

def show_video_soccer(data, fps=30):
    import imageio
    from IPython.display import Video, display 
    
    frames = [d['image'] for d in data]    
    frames_map = [d['map'] for d in data]    
    actions = [t['action'] for t in data]
    features = [state_features_soccer(**t) for t in data]
    distances = [np.linalg.norm(get_puck_center(t['soccer_state']) - cart_location(t['kart_info'])) for t in data]

    images = []
    map_images = []
    for frame, action, distance, feature in zip(frames, actions, distances, features):
        img = Image.fromarray(frame)        
        image_to_edit = ImageDraw.Draw(img)        
        image_to_edit.text((10, 10), "steering: {}".format(action.steer))         
        image_to_edit.text((10, 20), "drift: {}".format(action.drift))         
        image_to_edit.text((10, 30), "distance: {}".format(distance))         
        image_to_edit.text((10, 40), "angle diff: {}".format(feature[35]))         
        images.append(np.array(img))

    for img, action, distance, feature in zip(frames_map, actions, distances, features):
        image_to_edit = ImageDraw.Draw(img)        
        image_to_edit.text((10, 10), "steering: {}".format(action.steer), fill=(0, 0, 0))         
        image_to_edit.text((10, 20), "drift: {}".format(action.drift), fill=(0, 0, 0))  
        image_to_edit.text((10, 30), "distance: {}".format(distance), fill=(0, 0, 0))    
        image_to_edit.text((10, 40), "angle diff: {}".format(feature[35]), fill=(0, 0, 0))                     
        map_images.append(np.array(img))

    # create map video

    imageio.mimwrite('/tmp/test.mp4', images, fps=fps, bitrate=1000000)
    imageio.mimwrite('/tmp/test2.mp4', map_images, fps=fps, bitrate=1000000)
    display(Video('/tmp/test.mp4', width=800, height=600, embed=True))
    display(Video('/tmp/test2.mp4', width=800, height=600, embed=True))


def show_graph(data):

    steer = [t['action'].steer for t in data]
    drift = [t['action'].drift for t in data]
    fig, (steering_p, drift_p) = plt.subplots(1, 2)
    steering_p.plot(steer)
    steering_p.set_title("Steering")
    drift_p.plot(drift)
    drift_p.set_title("Drift")
    fig.show()

def show_trajectory_histogram(trajectories, metric=cart_overall_distance, min=0, max=1000, bins=10):
    # histogram of trajectory overall scoring distribution
    scores = [metric(t[-1]["kart_info"]) for t in trajectories]
    scores = np.clip(scores, min, max)
    plt.hist(scores, range(min, max, (max - min) // bins), density=True)
    plt.show()

viz_rollout = Rollout.remote(400, 300)
def run_agent(agent, n_steps=600, rollout=viz_rollout, **kwargs):
    data = ray.get(rollout.__call__.remote(agent, n_steps=n_steps, **kwargs))
    show_video(data)
    show_graph(data)
    return data

viz_rollout_soccer = Rollout.remote(400, 300, mode="soccer")
def run_soccer_agent(agent, rollout=viz_rollout_soccer, **kwargs):
    data = ray.get(rollout.__call__.remote(agent, **kwargs))
    show_video_soccer(data)
    show_graph(data)
    return data

last_mode = "track"
viz_rollouts = [Rollout.remote(50, 50, hd=False, render=False, frame_skip=5) for i in range(10)]
def rollout_many(many_agents, mode="track", **kwargs):    
    global viz_rollouts, last_mode
    if last_mode != mode:
        del viz_rollouts
        viz_rollouts = [Rollout.remote(50, 50, hd=False, render=False, frame_skip=5, mode=mode) for i in range(10)]
        last_mode = mode
    ray_data = []
    for i, agent in enumerate(many_agents):
         ray_data.append(viz_rollouts[i % len(viz_rollouts)].__call__.remote(agent, **kwargs) )    
    return ray.get(ray_data)

def dummy_agent(**kwargs):
    action = pystk.Action()
    action.acceleration = 1
    return action

class RaceTrackReinforcementConfiguration:

    mode = "track"

    def __init__(self) -> None:
        self.evaluator = OverallDistanceObjective()
        self.extractor = state_features
        self.agent = Agent
        self.training_agent = TrainingAgent

class SoccerReinforcementConfiguration(RaceTrackReinforcementConfiguration):

    mode = "soccer"

    def __init__(self) -> None:
        super().__init__()
        self.evaluator = SoccerBallDistanceObjective()
        self.extractor = state_features_soccer