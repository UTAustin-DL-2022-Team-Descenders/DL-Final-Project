import numpy as np
import torch

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
    
def cart_location(kart_info):
    # cart location
    return np.array(kart_info.location)[[0,2]].astype(np.float32)

def cart_front(kart_info):
    # cart front location
    return np.array(kart_info.front)[[0,2]].astype(np.float32)    

def cart_direction(kart_info):
    p = cart_location(kart_info)
    t = cart_front(kart_info)
    d = (p - t) / np.linalg.norm(p - t)
    return d

def cart_lateral_distances(kart_info, points):
    p = cart_location(kart_info)
    mid_point = points[:,1]
    dist = p - mid_point
    ortho_track = points[:,2] - points[:,1]
    ortho_track_u = ortho_track / np.linalg.norm(ortho_track)
    lat_dist = [np.dot(d, o) for d, o in zip(dist, ortho_track_u)]
    return np.array(lat_dist)

def cart_lateral_distance(kart_info, points):
    return cart_lateral_distances(kart_info, np.expand_dims(points, 0))[0]
    

def state_features(track_info, kart_info, absolute=False, **kwargs):
    
    # generates 5, 3, 2 tensor that contains the track geometry points x units down the track
    points = [three_points_on_track(kart_info.distance_down_track + d, track_info) for d in [0,5,10,15,20]]
    
    f = np.concatenate(points)
    if absolute:
        return f
    
    # cart location
    p = cart_location(kart_info)
    
    # cart direction unit vector    
    d = cart_direction(kart_info)

    # lane points relative to the kart location
    f = f - p[None]
    
    # (negated) cart orthogonal direction unit vector
    d_o = np.array([-d[1], d[0]], dtype=np.float32)
    
    laterals = np.concatenate([cart_lateral_distances(kart_info, np.array(points)), np.zeros(10)]).astype(np.float32)
   
    #print(laterals)

    return np.stack([f.dot(d), f.dot(d_o), laterals], axis=1)