import copy
import torch
import numpy as np
from torch.distributions import Bernoulli, Normal
from utils.utils import rollout_many, device
from utils.rewards import ObjectiveEvaluator, OverallDistanceObjective
from utils.track import state_features, three_points_on_track, cart_lateral_distance, get_obj1_to_obj2_angle, cart_location, cart_angle, get_puck_center
from utils.actors import Agent, TrainingAgent, SteeringActor

def collect_dist(trajectories):
    results = []
    for trajectory in trajectories:
        results.append(trajectory[-1]['kart_info'].overall_distance)
    return np.min(np.array(results)), np.max(np.array(results)), np.median(np.array(results))

overall_distance_objective = OverallDistanceObjective()

def reinforce(actor, 
              actors,  
              evaluator: ObjectiveEvaluator = overall_distance_objective,
              extractor = state_features,
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
        best_actor = actor.copy(best_action_net)
        assert(best_actor.train == actor.train)
        assert(best_actor.reward_type == actor.reward_type)

        trajectories = rollout_many([TrainingAgent(*slice_net, actor)]*n_trajectories, n_steps=600)
        
        # Compute all the reqired quantities to update the policy
        features = []
        returns = []
        actions = []
        losses = []
        
        # state features
        for trajectory in trajectories:
            for i in range(len(trajectory)):
                # Compute the features                
                state = extractor(**trajectory[i])
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

                current_lat = next_lat = None
                current_distance = next_distance = None
                if track_info:
                    points = three_points_on_track(kart_info.distance_down_track, track_info)
                    current_lat = cart_lateral_distance(kart_info, points)
                    current_distance = kart_info.overall_distance
                # current angle is where the kart is facing
                current_angle = cart_angle(kart_info)
                    
                kart_info = trajectory[min(i+T, len(trajectory)-1)]['kart_info']
                track_info = trajectory[min(i+T, len(trajectory)-1)]['track_info']
                if track_info:
                    points = three_points_on_track(kart_info.distance_down_track, track_info)
                    next_lat = cart_lateral_distance(kart_info, points)
                    next_distance = kart_info.overall_distance
                    # next angle is where the kart should be facing down the track (midpoint)
                    next_angle = get_obj1_to_obj2_angle(cart_location(kart_info), points[1])
                else:
                    soccer_state = trajectory[min(i+T, len(trajectory)-1)]['soccer_state']
                    next_angle = get_obj1_to_obj2_angle(cart_location(kart_info), get_puck_center(soccer_state))

                action = trajectory[i]['action']

                reward = actor.reward(
                                        action,
                                        current_lat=current_lat,
                                        next_lat=next_lat,
                                        current_distance=current_distance,
                                        next_distance=next_distance,
                                        current_angle=current_angle,
                                        next_angle=next_angle
                                     )
                
                loss.append(np.abs(current_lat))
            
                returns.append(reward) 
                
                # Store the action that we took
                actions.append( 
                    actor.extract_greedy_action(action)
                )

            it += len(trajectory)
            losses.append(np.sum(loss))
        
        # Upload everything to the GPU
        returns = torch.as_tensor(returns, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        features = torch.stack(features)

        print(returns)
        
        # enable this if not using discrete rewards! 
        #returns = (returns - returns.mean()) / returns.std()
        
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
            

        action_net.eval()
            
        best_performance = rollout_many([Agent(*slice_net, best_actor)] * n_validations, n_steps=600)
        current_performance = rollout_many([Agent(*slice_net, new_actor)] * n_validations, n_steps=600)
        
        # compute mean performance
        best_dist = evaluator.reduce(best_performance)
        dist = evaluator.reduce(current_performance)

        print('epoch = %d loss %d, dist = %s, best_dist = %s '%(epoch, np.abs(np.median(losses)), dist, best_dist))
        
        if evaluator.is_better_than(dist, best_dist):
            best_actor.action_net = best_action_net = copy.deepcopy(action_net)             
    return best_action_net