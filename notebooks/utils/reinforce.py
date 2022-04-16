import copy
import torch
import numpy as np
from torch.distributions import Bernoulli, Normal
from utils.utils import rollout_many, device
from utils.track import state_features, three_points_on_track, cart_lateral_distance
from utils.actors import Actor, GreedyActor, SteeringActor

def collect_dist(trajectories):
    results = []
    for trajectory in trajectories:
        results.append(trajectory[-1]['kart_info'].overall_distance)
    return np.array(results).mean()

def reinforce(action_net, 
             n_epochs = 10,
            n_trajectories = 100,
            n_iterations =100,
            n_validations = 20,
            batch_size = 128,
            T = 20):

    best_action_net = copy.deepcopy(action_net)

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)


    for epoch in range(n_epochs):
        eps = 1e-2
        
        # Roll out the policy, compute the Expectation
        trajectories = rollout_many([Actor(SteeringActor(action_net))]*n_trajectories, n_steps=600)
        
        # Compute all the reqired quantities to update the policy
        features = []
        returns = []
        actions = []
        loss = []

        # state features
        for trajectory in trajectories:
            for i in range(len(trajectory)):
                # Compute the features                
                state = state_features(**trajectory[i])
                features.append( torch.as_tensor(state, dtype=torch.float32).view(-1) )

        it = 0

        for trajectory in trajectories:
            for i in range(len(trajectory)):                
                lateral_sum = 0
                
                # Compute the returns

                # if the kart is off center too much
                kart_info = trajectory[i]['kart_info']
                track_info = trajectory[i]['track_info'] 
                points = three_points_on_track(kart_info.distance_down_track, track_info)
                current_lat = cart_lateral_distance(kart_info, points)
                    
                kart_info = trajectory[min(i+T, len(trajectory)-1)]['kart_info']
                track_info = trajectory[min(i+T, len(trajectory)-1)]['track_info']
                points = three_points_on_track(kart_info.distance_down_track, track_info)
                next_lat = cart_lateral_distance(kart_info, points)

                reward = 0
                
                loss.append(next_lat)

                # lateral distance reward
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
            
                #lateral_distance = state20[0,2]
                #print(state20, state)
                #print(lateral_distance)
                returns.append(reward)
                #returns.append(overall_distance)
                # Store the action that we took
                actions.append( trajectory[i]['action'].steer > 0 )

            it += len(trajectory)
        
        # Upload everything to the GPU
        returns = torch.as_tensor(returns, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        features = torch.stack(features)

        print(returns)
        
        returns = (returns - returns.mean()) / returns.std()
        
        action_net.train()
        avg_expected_log_return = []
        for it in range(n_iterations):
            batch_ids = torch.randint(0, len(returns), (batch_size,), device=device)
            batch_returns = returns[batch_ids]
            batch_actions = actions[batch_ids]
            batch_features = features[batch_ids]
            
            output = action_net(batch_features)
            pi = Bernoulli(probs=output[:,0])
            
            expected_log_return = (pi.log_prob(batch_actions)*batch_returns).mean()
            optim.zero_grad()
            (-expected_log_return).backward()
            optim.step()
            avg_expected_log_return.append(float(expected_log_return))
            
        best_performance = rollout_many([GreedyActor(SteeringActor(best_action_net))] * n_validations, n_steps=600)
        current_performance = rollout_many([GreedyActor(SteeringActor(action_net))] * n_validations, n_steps=600)
        
        # compute mean performance
        best_dist = collect_dist(best_performance)
        dist = collect_dist(current_performance)

        print('epoch = %d loss %d, dist = %d, best_dist = %d '%(epoch, np.abs(np.sum(loss)), dist, best_dist))
        
        if best_dist < dist:
            best_action_net = copy.deepcopy(action_net)

    return best_action_net