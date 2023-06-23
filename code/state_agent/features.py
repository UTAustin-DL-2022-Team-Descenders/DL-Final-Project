# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

from typing import List, Optional
import numpy as np
import torch

MAX_SPEED = 23.0
MIN_WALL_SPEED = 0.25
TARGET_SPEED_FEATURE = 44
PUCK_RADIUS = 2.0
PUCK_RADIUS_TIGHT = 0.75
PUCK_MAX_STEER_OFFSET = 0.45 # 35~45 degrees when [0,1] maps to [0,np.pi]
NEAR_WALL_OFFSET = 60.0
NEAR_WALL_STD = 2.0

PLANNER_BIAS = [0.0, 0.0, 0.0]

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
    return calculate_speed(kart_info, vel)

def get_puck_speed(puck_state):
    return puck_state.velocity

def calculate_speed(kart_info, vel):
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
  return np.array(puck_state.goal_line[team_id], dtype=np.float32)[:, [0, 2]]

def get_distance_cart_to_puck(kart_info, soccer_state):
    return np.linalg.norm(cart_location(kart_info) - get_puck_center(soccer_state))
@torch.jit.script
class SoccerFeatures:
    
    def __init__(self, features: Optional[torch.Tensor], planner_bias: Optional[torch.Tensor]):

        # Torch script doesn't like class instances !!!!
        self.PLAYER_PUCK_DISTANCE = 2
        self.PLAYER_WALL_DISTANCE = 3
        self.PLAYER_GOAL_DISTANCE = 4
        self.PUCK_GOAL_DISTANCE = 5
        self.PLANNER_CHOICE = 6
        self.LAST_PLANNER_CHOICE = 7
        self.PLANNER_CHOICE_COUNT = 8
        self.PLAYER_CENTER_ORIENTATION = 27
        self.PLAYER_REVERSE_STEER_ANGLE = 28
        self.PREVIOUS_SPEED = 29
        self.DELTA_SPEED_BEHIND = 30
        self.TARGET_SPEED_BEHIND = 31
        self.SPEED = 32
        self.TARGET_SPEED = 33
        self.DELTA_SPEED = 34
        self.PREVIOUS_ACCEL = 35
        self.PREVIOUS_STEER = 36
        self.PLAYER_PUCK_ATTACK_ANGLE = 37
        self.PUCK_GOAL_ANGLE = 38
        self.PLAYER_GOAL_ANGLE = 39
        self.PLAYER_PUCK_COUNTER_STEER_ANGLE = 40
        self.STEERING_ANGLE_BEHIND = 41
        self.PLAYER_PUCK_ANGLE = 42
        self.STEERING_ANGLE = 43
        self.PLAYER_PUCK_GOAL_ANGLE = 44

        self.features: torch.Tensor = features if features is not None else torch.zeros(size=[45])
        self.planner_bias: torch.Tensor = planner_bias if planner_bias is not None else torch.zeros(size=[3])

    def set_features(self, indices: List[int], values: torch.Tensor):
        for idx, f in zip(indices, values):
            self.features[idx] = f

    def select_indicies(self, indices: List[int]):
        return self.features[indices]

    def select_player_puck_goal_angle(self):
        return self.features[self.PLAYER_PUCK_GOAL_ANGLE]

    def select_player_goal_angle(self):
        return self.features[self.PLAYER_GOAL_ANGLE]

    def select_player_puck_angle(self):
        return self.features[self.PLAYER_PUCK_ANGLE]

    def select_player_puck_attack_angle(self):
        return self.features[self.PLAYER_PUCK_ATTACK_ANGLE]

    def select_steering_angle(self):
        return self.features[self.STEERING_ANGLE]

    def select_puck_goal_angle(self):
        return self.features[self.PUCK_GOAL_ANGLE]

    def select_behind_player_angle(self):
        return self.features[self.STEERING_ANGLE_BEHIND]

    def select_player_puck_countersteer_angle(self):
        return self.features[self.PLAYER_PUCK_COUNTER_STEER_ANGLE]

    def select_lateral_distance(self):
        return 0

    def select_speed(self):
        return self.features[self.SPEED]
    
    def select_speed_behind(self):
        return self.features[self.TARGET_SPEED_BEHIND]

    def select_delta_speed(self):
        return self.features[self.DELTA_SPEED]

    def select_delta_speed_behind(self):
        return self.features[self.DELTA_SPEED_BEHIND]

    def select_target_speed(self):
        return self.features[self.TARGET_SPEED]

    def select_player_puck_distance(self):
        return self.features[self.PLAYER_PUCK_DISTANCE]

    def select_puck_goal_distance(self):
        return self.features[self.PUCK_GOAL_DISTANCE]

    def select_player_reverse_steer_angle(self):
        return self.features[self.PLAYER_REVERSE_STEER_ANGLE]

    def select_planner_choice(self):
        return self.features[self.PLANNER_CHOICE]

    def select_prev_planner_choice(self):
        return self.features[self.LAST_PLANNER_CHOICE]

    def select_planner_count(self):
        # the number of times the planner has chosen this policy action in a row
        return self.features[self.PLANNER_CHOICE_COUNT]

    def selection_planner_bias(self):
        return self.planner_bias

    def select_center_orientation(self):
        return self.features[self.PLAYER_CENTER_ORIENTATION]

# use this as a replacement for the class instance to grab the feature labels
SoccerFeaturesLabels = SoccerFeatures(None, None) # type: ignore

def extract_all_features(kart_info, soccer_state, team_num, absolute=False, target_speed=0.0, last_state=None, last_action=None, last_feature:Optional[SoccerFeatures]=None, **kwargs):

    # cart location
    p = cart_location(kart_info)

    # cart front
    front = cart_front(kart_info)

    # puck
    puck = get_puck_center(soccer_state)

    # goal
    team_num = 1 - team_num
    goal_line = get_team_goal_line(soccer_state, team_num)
    goal = get_team_goal_line_center(soccer_state, team_num)
    goal_line_length = np.linalg.norm(goal_line)
    goal_line_length_margin = goal_line_length - PUCK_RADIUS * 4

    # orientation
    vec_orientation = front - p
    vec_orientation /= np.linalg.norm(vec_orientation + 0.00001)
    vec_center = - p
    vec_center /= np.linalg.norm(vec_center + 0.00001)
    center_orientation = np.dot(vec_orientation, vec_center.T)

    # vectors
    vec_puck_p = p - puck
    vec_puck_p /= np.linalg.norm(vec_puck_p + 0.00001)
    vec_puck_goal = puck - goal
    vec_puck_goal /= np.linalg.norm(vec_puck_goal + 0.00001)
    vec_goal_line = goal_line[1] - goal
    vec_goal_line /= np.linalg.norm(vec_goal_line + 0.00001)

    target_point = (PUCK_RADIUS_TIGHT * vec_puck_goal) + puck

    # closest goal point + puck radius
    #proj_goal_dist = np.dot(vec_goal_line, vec_puck_goal.T)
    #proj_goal_dist = np.clip(proj_goal_dist, -goal_line_length_margin/2, goal_line_length_margin/2)
    #goal = proj_goal_dist * vec_goal_line + goal

    # recalculate the puck-goal vector
    #vec_puck_goal = puck - goal
    #vec_puck_goal /= np.linalg.norm(vec_puck_goal + 0.00001)

    # steering angles
    steer_angle = get_obj1_to_obj2_angle(p, front)
    steer_angle_behind = get_obj1_to_obj2_angle(p, front) + np.pi
    steer_angle_puck = get_obj1_to_obj2_angle(front, puck)
    steer_angle_goal = get_obj1_to_obj2_angle(p, goal)
    steer_angle_puck_goal = get_obj1_to_obj2_angle(puck, goal)
    steer_angle_goal_diff = get_obj1_to_obj2_angle_difference(steer_angle, steer_angle_goal)
    steer_puck_angle_diff = get_obj1_to_obj2_angle_difference(steer_angle, steer_angle_puck)
    steer_angle_puck_incidence = get_obj1_to_obj2_angle(p, target_point)
    steer_angle_puck_incidence_diff = get_obj1_to_obj2_angle_difference(steer_angle, steer_angle_puck_incidence)
    #steer_puck_goal_angle_diff = steer_angle_goal_diff - steer_puck_angle_diff

    # counter steering up to 30 degrees (1/6 pi)
    #steer_angle_puck_goal_counter_steer = -1 * np.sign(steer_puck_angle_diff) * (np.clip(steer_puck_goal_angle_diff, -0.15, 0.15)) + steer_puck_angle_diff
    steer_angle_puck_goal_counter_steer = limit_period(steer_puck_angle_diff - np.clip(steer_angle_goal_diff, -PUCK_MAX_STEER_OFFSET, PUCK_MAX_STEER_OFFSET))
    #get_obj1_to_obj2_angle_difference(steer_angle, steer_angle_puck)  #steer_puck_angle_diff #np.sign(steer_puck_angle_diff) * np.clip(np.abs(steer_puck_angle_diff), 0, 1.0/100.0)

    # puck angle of incidence towards the goal
    # is the puck currently between us and the goal
    puck_facing_towards_goal = np.dot(vec_puck_p, vec_puck_goal.T) > 0

    # distance
    pp_dist = np.linalg.norm(p - puck) - PUCK_RADIUS

    # speed
    speed = cart_speed(kart_info)
    speed_negative = -10
    previous_speed = cart_speed(last_state[-1]) if last_state else MAX_SPEED

    features = np.zeros(45).astype(np.float32)

    features[0:2] = p - puck
    features[SoccerFeaturesLabels.PLAYER_WALL_DISTANCE] = (np.linalg.norm(p) - NEAR_WALL_OFFSET) / (NEAR_WALL_STD)
    features[SoccerFeaturesLabels.PLAYER_PUCK_DISTANCE] =  pp_dist
    features[SoccerFeaturesLabels.PLAYER_GOAL_DISTANCE] = np.linalg.norm(p - goal)
    features[SoccerFeaturesLabels.PUCK_GOAL_DISTANCE] = np.linalg.norm(puck - goal)
    features[SoccerFeaturesLabels.SPEED] = speed
    #features[SoccerFeaturesLabels.PUCK_SPEED] = get_puck_speed(puck_state=soccer_state)
    features[SoccerFeaturesLabels.PLAYER_CENTER_ORIENTATION] = center_orientation
    features[SoccerFeaturesLabels.PREVIOUS_ACCEL] = last_action.acceleration if last_action is not None else 0.0
    features[SoccerFeaturesLabels.PREVIOUS_STEER] = last_action.steer if last_action is not None else 0.0
    features[SoccerFeaturesLabels.PREVIOUS_SPEED] = previous_speed
    features[SoccerFeaturesLabels.TARGET_SPEED] = target_speed
    features[SoccerFeaturesLabels.TARGET_SPEED_BEHIND] = speed_negative
    features[SoccerFeaturesLabels.DELTA_SPEED] = target_speed - speed
    features[SoccerFeaturesLabels.DELTA_SPEED_BEHIND] = speed_negative - speed
    features[SoccerFeaturesLabels.STEERING_ANGLE] = steer_puck_angle_diff # by default, steer towards the puck
    features[SoccerFeaturesLabels.STEERING_ANGLE_BEHIND] = steer_angle_behind
    features[SoccerFeaturesLabels.PLAYER_PUCK_ATTACK_ANGLE] = steer_angle_puck_incidence_diff
    features[SoccerFeaturesLabels.PLAYER_GOAL_ANGLE] = steer_angle_goal_diff
    features[SoccerFeaturesLabels.PLAYER_PUCK_ANGLE] = steer_puck_angle_diff
    features[SoccerFeaturesLabels.PLAYER_PUCK_GOAL_ANGLE] = 0 #steer_puck_goal_angle_diff
    features[SoccerFeaturesLabels.PLAYER_PUCK_COUNTER_STEER_ANGLE] = steer_angle_puck_goal_counter_steer
    features[SoccerFeaturesLabels.PUCK_GOAL_ANGLE] = steer_angle_puck_goal
    features[SoccerFeaturesLabels.PLAYER_REVERSE_STEER_ANGLE] = -1.0 * np.sign(steer_puck_angle_diff)
    features[SoccerFeaturesLabels.LAST_PLANNER_CHOICE] = last_feature.select_planner_choice() if last_feature else -1
    features[SoccerFeaturesLabels.PLANNER_CHOICE_COUNT] = last_feature.select_planner_count() * (last_feature.select_planner_choice() == last_feature.select_prev_planner_choice()) + 1 if last_feature else 0

    return SoccerFeatures(torch.as_tensor(features), torch.as_tensor(PLANNER_BIAS))
