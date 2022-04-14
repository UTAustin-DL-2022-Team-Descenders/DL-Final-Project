import copy
import torch
import numpy as np
from torch.distributions import Bernoulli
from utils.utils import rollout_many, device
from utils.track import state_features
from utils.actors import Actor, GreedyActor

def reinforce(action_net, 
             n_epochs = 10,
            n_trajectories = 100,
            n_iterations =100,
            batch_size = 128,
            T = 20):

    best_action_net = copy.deepcopy(action_net)

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)


    for epoch in range(n_epochs):
        eps = 1e-2
        
        # Roll out the policy, compute the Expectation
        trajectories = rollout_many([Actor(action_net)]*n_trajectories, n_steps=600)
        
        # Compute all the reqired quantities to update the policy
        features = []
        returns = []
        actions = []
        for trajectory in trajectories:
            for i in range(len(trajectory)):
                state = state_features(**trajectory[i])
                state20 = state_features(**trajectory[min(i+T, len(trajectory)-1)])                        
                # Compute the returns
                overall_distance = trajectory[min(i+T, len(trajectory)-1)]['kart_info'].overall_distance - \
                    trajectory[i]['kart_info'].overall_distance
                #lateral_distance = state20[0,2]
                #print(state20, state)
                #print(lateral_distance)
                #returns.append(-(lateral_distance ** 2))
                returns.append(overall_distance)
                # Compute the features
                features.append( torch.as_tensor(state, dtype=torch.float32).view(-1) )
                # Store the action that we took
                actions.append( trajectory[i]['action'].steer > 0 )
        
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
            pi = Bernoulli(logits=output[:,0])
            
            expected_log_return = (pi.log_prob(batch_actions)*batch_returns).mean()
            optim.zero_grad()
            (-expected_log_return).backward()
            optim.step()
            avg_expected_log_return.append(float(expected_log_return))
            
        best_performance, current_performance = rollout_many([GreedyActor(best_action_net), GreedyActor(action_net)], n_steps=600)
        
        print('epoch = %d   best_dist = '%epoch, best_performance[-1]['kart_info'].overall_distance)
        
        if best_performance[-1]['kart_info'].overall_distance < current_performance[-1]['kart_info'].overall_distance:
            best_action_net = copy.deepcopy(action_net)

    return best_action_net