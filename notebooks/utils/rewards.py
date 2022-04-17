import numpy as np

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
            reward = 0
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
            reward = 0
    else:
        # strong reward
        reward = 2    
    return reward

def distance_traveled_reward(current_distance, next_distance, max=100):

    return np.clip(next_distance - current_distance, 0, max)
