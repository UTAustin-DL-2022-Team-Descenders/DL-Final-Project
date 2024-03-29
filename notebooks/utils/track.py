# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import numpy as np
import torch

MAX_SPEED = 23.0
TARGET_SPEED_FEATURE = 44

def three_points_on_track(distance, track):    
    distance = np.clip(distance, track.path_distance[0,0], track.path_distance[-1,1]).astype(np.float32)
    #print("Distance", distance)
    valid_node = (track.path_distance[..., 0] <= distance) & (distance <= track.path_distance[..., 1])
    valid_node_idx, = np.where(valid_node)
    node_idx = valid_node_idx[0] # np.random.choice(valid_node_idx)
    d = track.path_distance[node_idx].astype(np.float32)
    x = track.path_nodes[node_idx][:,[0,2]].astype(np.float32) # Ignore the y coordinate
    w, = track.path_width[node_idx].astype(np.float32)    
    t = (distance - d[0]) / (d[1] - d[0])
    mid = x[1] * t + x[0] * (1 - t)
    x10 = (x[1] - x[0]) / np.linalg.norm(x[1]-x[0])
    #print("Track dir", x10, "Index", node_idx)
    
    x10_ortho = np.array([-x10[1],x10[0]], dtype=np.float32)
    return mid - w / 2 * x10_ortho, mid, mid + w / 2 * x10_ortho
    
def get_obj1_to_obj2_angle(object1_center, object2_center):
    object1_direction = get_obj1_to_obj2_direction(object1_center, object2_center)
    return np.arctan2(object1_direction[1], object1_direction[0])

def get_obj1_to_obj2_direction(object1_center, object2_center):
    norm = np.linalg.norm(object2_center-object1_center)
    return (object2_center-object1_center) / (norm + 0.00001)

def get_object_center(state_dict):
  return np.array(state_dict.location, dtype=np.float32)[[0, 2]]

def get_puck_center(puck_state):
  return get_object_center(puck_state.ball)

# limit angle between -1 to 1
def limit_period(angle):
  return angle - np.floor(angle / 2 + 0.5) * 2 

def get_obj1_to_obj2_angle_difference(object1_angle, object2_angle):
  angle_difference = (object1_angle - object2_angle) / np.pi
  return angle_difference

def cart_location(kart_info):
    # cart location
    return np.array(kart_info.location)[[0,2]].astype(np.float32)

def cart_front(kart_info):
    # cart front location
    return np.array(kart_info.front)[[0,2]].astype(np.float32)    

def cart_direction(kart_info):
    p = cart_location(kart_info)
    t = cart_front(kart_info)
    d = (t - p) / np.linalg.norm(t - p)
    return d

def cart_angle(kart_info):
    p = cart_location(kart_info)
    front = cart_front(kart_info)    
    return get_obj1_to_obj2_angle(p, front)

def cart_lateral_distances(kart_info, points):
    p = cart_location(kart_info)
    mid_point = points[:,1]
    dist = p - mid_point
    ortho_track = points[:,2] - points[:,1]
    ortho_track_u = ortho_track / (np.linalg.norm(ortho_track) + 0.00001)
    lat_dist = [np.dot(d, o) for d, o in zip(dist, ortho_track_u)]
    return np.array(lat_dist)

def cart_lateral_distance(kart_info, points):
    return cart_lateral_distances(kart_info, np.expand_dims(points, 0))[0]
    
def cart_overall_distance(kart_info, **kwargs):
    return kart_info.overall_distance

def cart_velocity(kart_info):
    return np.array(kart_info.velocity)[[0,2]]

def cart_speed(kart_info):
    # cart speed
    vel = cart_velocity(kart_info)
    dir = cart_direction(kart_info)
    sign = np.sign(np.dot(vel, dir))
    speed = np.linalg.norm(vel) * sign
    return speed

def get_target_speed_feature(features):
    return features[TARGET_SPEED_FEATURE]

class Features():
    pass

class SoccerFeatures(Features):
    
    DELTA_STEERING_ANGLE = 44
    DELTA_SPEED = 40
    TARGET_SPEED = 39
    
    def get_feature_vector(self, track_info, kart_info, soccer_state, absolute=False, target_speed=0.0, **kwargs):

        # cart location
        p = cart_location(kart_info)

        # cart front
        front = cart_front(kart_info)

        # puck
        puck = get_puck_center(soccer_state)
        
        # steering angles to points down the track
        steer_angle = get_obj1_to_obj2_angle(p, front)
        steer_angle_puck = get_obj1_to_obj2_angle(p, puck)
        
        # speed
        speed = cart_speed(kart_info)

        features = np.zeros(45).astype(np.float32)

        features[0:2] = p - puck
        features[self.TARGET_SPEED] = target_speed
        features[self.DELTA_SPEED] = target_speed - speed
        features[self.DELTA_STEERING_ANGLE] = get_obj1_to_obj2_angle_difference(steer_angle, steer_angle_puck)

        return features

    def select_delta_steering(self, features):        
        return features[self.DELTA_STEERING_ANGLE]

    def select_lateral_distance(self, features):
        return 0

    def select_delta_speed(self, features):
        return features[self.DELTA_SPEED]

    def select_target_speed(self, features):
        return features[self.TARGET_SPEED]


class TrackFeatures(Features):
    
    DELTA_STEERING_ANGLE = 44

    def get_feature_vector(self, track_info, kart_info, absolute=False, target_speed=0.0, **kwargs):
    
        # generates 5, 3, 2 tensor that contains the track geometry points x units down the track
        points = [three_points_on_track(kart_info.distance_down_track + d, track_info) for d in [0,5,10,15,20]]
        
        f = np.concatenate(points)
        if absolute:
            return f
        
        # cart location
        p = cart_location(kart_info)

        # cart front
        front = cart_front(kart_info)
        
        # cart direction unit vector    
        d = cart_direction(kart_info)

        # lane points relative to the kart location
        f = f - p[None]
        
        # (negated) cart orthogonal direction unit vector
        d_o = np.array([-d[1], d[0]], dtype=np.float32)
        
        # steering angles to points down the track
        steer_angle = get_obj1_to_obj2_angle(p, front)
        steering_angles = [
            get_obj1_to_obj2_angle_difference(steer_angle, get_obj1_to_obj2_angle(p, point[1])) 
            for point in points
        ]

        merged = np.concatenate([cart_lateral_distances(kart_info, np.array(points)), np.zeros(5), np.array(steering_angles)]).astype(np.float32)
    
        #print(laterals)

        return np.stack([f.dot(d), f.dot(d_o), merged], axis=1).flatten()

    def select_delta_steering(self, features):
        return features[self.DELTA_STEERING_ANGLE]

    def select_lateral_distance(self, features):
        return features[30]    
