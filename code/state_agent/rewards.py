# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import numpy as np
from .features import cart_location, get_puck_center, MAX_SPEED

MAX_STEERING_ANGLE_REWARD = np.pi
MAX_SPEED_REWARD = 3.0
MAX_SOCCER_DISTANCE_REWARD = 3.0
MAX_DISTANCE = 130

def lateral_distance_reward(current_lat, next_lat):
    # lateral distance reward
    reward = 0
    if np.abs(current_lat) > 1:

        # if the lateral distance shrinking?
        if np.abs(next_lat) < np.abs(current_lat):
            # less strong reward
            reward = 1
        else:
            # no reward
            reward = -1
    else:
        # strong reward
        reward = 2
    return reward

def lateral_distance_causal_reward(current_lat, next_lat):
    # lateral distance reward
    reward = 0
    if np.abs(next_lat) > 1:

        # if the lateral distance shrinking?
        if np.abs(next_lat) < np.abs(current_lat):
            # less strong reward
            reward = 1
        else:
            # no reward
            reward = -1
    else:
        # strong reward
        reward = 2
    return reward

def distance_traveled_reward(current_distance, next_distance, max=100):

    return np.clip(next_distance - current_distance, 0, max)

def continuous_causal_reward(current_, next_, threshold, max):

    reward = 0
    if np.abs(next_) > threshold:
        # is the shrinking?
        if np.abs(next_) < np.abs(current_):
            # less strong reward
            reward = max - np.abs(next_)
        else:
            # no reward
            reward = -1
    else:
        # strong reward
        reward = max

    return reward

def continuous_causal_reward_ext(current_, next_, threshold, threshold_shrink, max):

    reward = 0
    if np.abs(next_) > threshold:
        # is the shrinking?
        if np.abs(next_) + threshold_shrink < np.abs(current_):
            # less strong reward
            reward = max - np.abs(next_)
        else:
            # no reward
            reward = - (max - np.abs(next_))
    else:
        # strong reward
        reward = max

    return reward

def steering_angle_reward(current_angle, next_angle):

    (c_delta) = current_angle
    (n_delta) = next_angle
    multiplier = 1

    return multiplier * continuous_causal_reward(c_delta, n_delta, threshold=np.deg2rad(0.5), max=MAX_STEERING_ANGLE_REWARD)

def speed_reward(current_speed, next_speed):
    return continuous_causal_reward(current_speed / MAX_SPEED * MAX_SPEED_REWARD, next_speed / MAX_SPEED * MAX_SPEED_REWARD, threshold=0.5, max=MAX_SPEED_REWARD)


class ObjectiveEvaluator:

    def reduce(self, trajectories):
        pass

    def is_better_than(self, a, b):
        return a > b

class OverallDistanceObjective(ObjectiveEvaluator):

    def reduce(self, trajectories):
        results = []
        for trajectory in trajectories:
            results.append(trajectory[-1]['kart_info'].overall_distance)
        return np.min(np.array(results)), np.max(np.array(results)), np.median(np.array(results))

    def is_better_than(self, a, b):
        return super().is_better_than(a[2], b[2])

class TargetDistanceObjective(ObjectiveEvaluator):

    def __init__(self, max_distance=MAX_DISTANCE) -> None:
        super().__init__()
        self.max_distance = max_distance

    def set_target(self, target):
        self.target = target

    def get_target(self, state):
        # calculate the target position
        return self.target

    def calculate_state_score(self, t):
        ball = self.get_target(t)
        pos = cart_location(t['kart_info'])
        distance = 1.0 - (np.linalg.norm(pos - ball) / self.max_distance)
        return distance

    def calculate_trajectory_score(self, trajectory):

        # calculate the score as the normalized distance away from the target at each time step
        total = 0
        for t in trajectory:
            total += self.calculate_state_score(t)

        return total

    def reduce(self, trajectories):
        results = [self.calculate_trajectory_score(trajectory) for trajectory in trajectories]
        return np.min(np.array(results)), np.max(np.array(results)), np.median(np.array(results))

    def is_better_than(self, a, b):
        return super().is_better_than(a[2], b[2])

class SoccerBallDistanceObjective(TargetDistanceObjective):

    def get_target(self, t):
        return get_puck_center(t['soccer_state'])
