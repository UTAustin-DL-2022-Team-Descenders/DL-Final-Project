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
    return np.min(np.array(results)), np.max(np.array(results)), np.median(np.array(results))

def reinforce(actor, 
              actors,  
             n_epochs = 10,
            n_trajectories = 100,
            n_iterations =100,
            n_validations = 20,
            batch_size = 128,
            T = 20):

    action_net = actor.action_net
    best_action_net = copy.deepcopy(action_net)

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)

    slice_net = list(filter(lambda a: a != actor, actors))

    for epoch in range(n_epochs):
        eps = 1e-2  
        
        # Roll out the policy, compute the Expectation
        assert(actor.action_net == action_net)
        trajectories = rollout_many([Actor(*slice_net, actor)]*n_trajectories, n_steps=600)
        
        # Compute all the reqired quantities to update the policy
        features = []
        returns = []
        actions = []
        losses = []
        
        # state features
        for trajectory in trajectories:
            for i in range(len(trajectory)):
                # Compute the features                
                state = state_features(**trajectory[i])
                features.append( torch.as_tensor(state, dtype=torch.float32).view(-1) )

        it = 0

        for trajectory in trajectories:

            loss = []

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

                reward = actor.reward(
                                        current_lat=current_lat,
                                        next_lat=next_lat
                                     )
                
                loss.append(current_lat)

            
                #lateral_distance = state20[0,2]
                #print(state20, state)
                #print(lateral_distance)
                returns.append(reward) 
                #returns.append(overall_distance)
                # Store the action that we took
                actions.append( trajectory[i]['action'].steer > 0 )

            it += len(trajectory)
            losses.append(np.sum(loss))
        
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
            

        new_actor = actor.__class__(action_net)
        best_actor = actor.__class__(best_action_net)
        
        best_performance = rollout_many([GreedyActor(*slice_net, best_actor)] * n_validations, n_steps=600)
        current_performance = rollout_many([GreedyActor(*slice_net, new_actor)] * n_validations, n_steps=600)
        
        # compute mean performance
        best_dist = collect_dist(best_performance)
        dist = collect_dist(current_performance)

        print('epoch = %d loss %d, dist = %s, best_dist = %s '%(epoch, np.abs(np.median(losses)), dist, best_dist))
        
        if best_dist[2] < dist[2]:
            best_action_net = copy.deepcopy(action_net)
            actor = new_actor
        
    return best_action_net