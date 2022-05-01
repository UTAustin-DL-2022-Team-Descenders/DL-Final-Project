# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import copy
import torch
import numpy as np
from torch.distributions import Bernoulli, Normal
from utils.utils import rollout_many, device, RaceTrackReinforcementConfiguration

def collect_dist(trajectories):
    results = []
    for trajectory in trajectories:
        results.append(trajectory[-1]['kart_info'].overall_distance)
    return np.min(np.array(results)), np.max(np.array(results)), np.median(np.array(results))

race_track_configuration = RaceTrackReinforcementConfiguration()

def reinforce(actor, 
              actors,  
              configuration: RaceTrackReinforcementConfiguration = race_track_configuration,              
             n_epochs = 10,
            n_trajectories = 100,
            n_iterations =100,
            n_validations = 20,
            n_steps=600,
            batch_size = 128,
            T = 20):

    action_net = actor.action_net
    best_action_net = copy.deepcopy(action_net)
    best_actor = actor.copy(best_action_net)
    evaluator = configuration.evaluator

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)

    slice_net = list(filter(lambda a: a != actor, actors))
    
    for epoch in range(n_epochs):
        eps = 1e-2  
        
        # Roll out the policy, compute the Expectation
        assert(actor.action_net == action_net)        
        assert(best_actor.train == actor.train)
        assert(best_actor.reward_type == actor.reward_type)

        agents = [configuration.agent(*slice_net, actor, train=True) for i in range(n_trajectories)]
        trajectories = rollout_many(agents, mode=configuration.mode, randomize=True, n_steps=n_steps)
        
        # Compute all the reqired quantities to update the policy
        features = []
        returns = []
        actions = []
        losses = []
        
        # state features
        for num, trajectory in enumerate(trajectories):
            for i in range(len(trajectory)):
                # Compute the features                
                state = actor.select_features(agents[num].extractor, agents[num].get_feature_vector(**trajectory[i]))
                features.append( torch.as_tensor(state, dtype=torch.float32).view(-1) )

        it = 0

        for num, trajectory in enumerate(trajectories):

            loss = []

            for i in range(len(trajectory)):                
                # Compute the returns

                # if the kart is off center too much                  
                action = trajectory[i]['action']

                reward = actor.reward(
                    action,
                    agents[num].extractor,
                    features[it + i],
                    features[it + min(i + T, len(trajectory)-1)]
                )
                
                loss.append(0)
            
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
          
            output = action_net(batch_features, train="reinforce")
            pi = Bernoulli(probs=output)
            
            expected_log_return = (pi.log_prob(batch_actions).squeeze()*batch_returns).mean()
            optim.zero_grad()
            (-expected_log_return).backward()
            optim.step()
            avg_expected_log_return.append(float(expected_log_return))           
            

        action_net.eval()
            
        best_performance = rollout_many([configuration.agent(*slice_net, best_actor) for i in range(n_validations)], mode=configuration.mode, n_steps=n_steps)
        current_performance = rollout_many([configuration.agent(*slice_net, actor) for i in range(n_validations)], mode=configuration.mode, n_steps=n_steps)
        
        # compute mean performance
        best_dist = evaluator.reduce(best_performance)
        dist = evaluator.reduce(current_performance)

        print('epoch = %d loss %d, dist = %s, best_dist = %s '%(epoch, np.abs(np.median(losses)), dist, best_dist))
        
        if evaluator.is_better_than(dist, best_dist):
            best_actor.action_net = best_action_net = copy.deepcopy(action_net)             
    return best_action_net