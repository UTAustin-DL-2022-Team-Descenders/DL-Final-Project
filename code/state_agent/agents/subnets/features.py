# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import numpy as np
import torch

MAX_SPEED = 23.0
TARGET_SPEED_FEATURE = 44
PUCK_RADIUS = 2.0
PUCK_MAX_STEER_OFFSET = 0.2 # 35~45 degrees when [0,1] maps to [0,np.pi]
    
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
  return limit_period(angle_difference)

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

def get_team_goal_line_center(puck_state, team_id):
  value = np.array(puck_state.goal_line[team_id], dtype=np.float32)  
  return value[:, [0, 2]].mean(axis=0)

def get_team_goal_line(puck_state, team_id):
  return torch.tensor(puck_state.goal_line[team_id], dtype=torch.float32)[:, [0, 2]]

def get_distance_cart_to_puck(kart_info, soccer_state):
    return np.linalg.norm(cart_location(kart_info) - get_puck_center(soccer_state))

class Features():
    pass

class SoccerFeatures(Features):
    
    PLAYER_PUCK_DISTANCE = 2
    PUCK_GOAL_DISTANCE = 5    
    DELTA_SPEED_BEHIND = 30
    TARGET_SPEED_BEHIND = 31
    SPEED = 32
    TARGET_SPEED = 33    
    DELTA_SPEED = 34
    PLAYER_GOAL_ANGLE = 39
    PLAYER_PUCK_COUNTER_STEER_ANGLE = 40        
    STEERING_ANGLE_BEHIND = 41
    PLAYER_PUCK_ANGLE = 42
    STEERING_ANGLE = 43
    PLAYER_PUCK_GOAL_ANGLE = 44
    
    def get_feature_vector(self, kart_info, soccer_state, absolute=False, target_speed=0.0, **kwargs):

        # cart location
        p = cart_location(kart_info)

        # cart front
        front = cart_front(kart_info)

        # puck
        puck = get_puck_center(soccer_state)

        # goal
        goal = get_team_goal_line_center(soccer_state, 1) # team is hard-coded!!!!

        # steering angles
        steer_angle = get_obj1_to_obj2_angle(p, front)
        steer_angle_behind = get_obj1_to_obj2_angle(p, front) + np.pi
        steer_angle_puck = get_obj1_to_obj2_angle(front, puck)
        steer_angle_goal = get_obj1_to_obj2_angle(p, goal)
        steer_angle_puck_goal = get_obj1_to_obj2_angle(puck, goal)                
        steer_angle_goal_diff = get_obj1_to_obj2_angle_difference(steer_angle, steer_angle_goal)
        steer_puck_angle_diff = get_obj1_to_obj2_angle_difference(steer_angle, steer_angle_puck)
        #steer_puck_goal_angle_diff = steer_angle_goal_diff - steer_puck_angle_diff

        # counter steering up to 30 degrees (1/6 pi)
        #steer_angle_puck_goal_counter_steer = -1 * np.sign(steer_puck_angle_diff) * (np.clip(steer_puck_goal_angle_diff, -0.15, 0.15)) + steer_puck_angle_diff
        steer_angle_puck_goal_counter_steer = limit_period(steer_puck_angle_diff - np.clip(steer_angle_goal_diff, -PUCK_MAX_STEER_OFFSET, PUCK_MAX_STEER_OFFSET))
        #get_obj1_to_obj2_angle_difference(steer_angle, steer_angle_puck)  #steer_puck_angle_diff #np.sign(steer_puck_angle_diff) * np.clip(np.abs(steer_puck_angle_diff), 0, 1.0/100.0)
        
        # distance 
        pp_dist = np.linalg.norm(p - puck) - PUCK_RADIUS
        
        # speed
        speed = cart_speed(kart_info)
        speed_negative = -10

        features = np.zeros(45).astype(np.float32)

        features[0:2] = p - puck
        features[self.PLAYER_PUCK_DISTANCE] =  pp_dist
        features[self.PUCK_GOAL_DISTANCE] = np.linalg.norm(puck - goal)
        features[self.SPEED] = speed
        features[self.TARGET_SPEED] = target_speed
        features[self.TARGET_SPEED_BEHIND] = speed_negative
        features[self.DELTA_SPEED] = target_speed - speed
        features[self.DELTA_SPEED_BEHIND] = speed_negative - speed
        features[self.STEERING_ANGLE_BEHIND] = steer_angle_behind
        features[self.PLAYER_GOAL_ANGLE] = steer_angle_goal_diff
        features[self.PLAYER_PUCK_ANGLE] = steer_puck_angle_diff
        features[self.PLAYER_PUCK_GOAL_ANGLE] = 0 #steer_puck_goal_angle_diff
        features[self.PLAYER_PUCK_COUNTER_STEER_ANGLE] = steer_angle_puck_goal_counter_steer

        return features

    def select_indicies(self, indices, features):
        return torch.Tensor(features[indices])

    def select_player_puck_goal_angle(self, features):        
        return features[self.PLAYER_PUCK_GOAL_ANGLE]

    def select_player_goal_angle(self, features):        
        return features[self.PLAYER_GOAL_ANGLE]

    def select_player_puck_angle(self, features):        
        return features[self.PLAYER_PUCK_ANGLE]

    def select_behind_player_angle(self, features):        
        return features[self.STEERING_ANGLE_BEHIND]

    def select_player_puck_countersteer_angle(self, features):
        return features[self.PLAYER_PUCK_COUNTER_STEER_ANGLE]

    def select_lateral_distance(self, features):
        return 0

    def select_speed(self, features):
        return features[self.SPEED]

    def select_speed_behind(self, features):
        return features[self.TARGET_SPEED_BEHIND]

    def select_delta_speed(self, features):
        return features[self.DELTA_SPEED]

    def select_delta_speed_behind(self, features):
        return features[self.DELTA_SPEED_BEHIND]

    def select_target_speed(self, features):
        return features[self.TARGET_SPEED]

    def select_player_puck_distance(self, features):
        return features[self.PLAYER_PUCK_DISTANCE]

    def select_puck_goal_distance(self, features):
        return features[self.PUCK_GOAL_DISTANCE]