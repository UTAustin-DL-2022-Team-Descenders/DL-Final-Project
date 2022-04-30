# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import copy
import torch
import numpy as np
from torch.distributions import Bernoulli, Normal
from state_agent.agents.subnets.rewards import SoccerBallDistanceObjective
from state_agent.agents.subnets.utils import DictObj

def collect_dist(trajectories):
    results = []
    for trajectory in trajectories:
        results.append(trajectory[-1]['kart_info'].overall_distance)
    return np.min(np.array(results)), np.max(np.array(results)), np.median(np.array(results))

def raw_trajectory_to_player_trajectory(t, player_team, player_on_team):
    return {
        "kart_info": DictObj(t['team_state'][player_team][player_on_team]['kart']),
        "soccer_state": DictObj(t['soccer_state'])
    }

class SoccerReinforcementConfiguration:

    mode = "soccer"

    def __init__(self) -> None:
        super().__init__()        
        self.evaluator = SoccerBallDistanceObjective()        

configuration = SoccerReinforcementConfiguration()

def reinforce(actor, 
              agent,  
              configuration: SoccerReinforcementConfiguration = configuration,              
                n_epochs = 10,            
                n_iterations =100,                         
                batch_size = 128,
                T = 20
):
    
    for epoch in range(n_epochs):
        reinforce_epoch(
            agent, actor, 
            trajectories,
            epoch=epoch,
            configuration=configuration,             
            batch_size=batch_size,
            iterations=n_iterations,
        )

    return best_action_net

def reinforce_epoch(
    agent,
    actor,    
    trajectories,  
    epoch=0,        
    configuration=configuration,
    iterations=100,      
    batch_size=128,
    reward_window=1,
    context=None  
):

    action_net = actor.action_net
    best_action_net = copy.deepcopy(action_net)
    best_actor = actor.copy(best_action_net)
    evaluator = configuration.evaluator

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)
    
    T = reward_window
    eps = 1e-2  
        
    # Roll out the policy, compute the Expectation
    assert(actor.action_net == action_net)        
    assert(best_actor.train == actor.train)
    assert(best_actor.reward_type == actor.reward_type)

    # Compute all the reqired quantities to update the policy
    features = []
    returns = []
    actions = []
    losses = []

    # XXX change these
    player_team = 0 # hard-coded for now
    player_on_team = 0 # hard-coded for now               
    player_id = 0 # hard-coded for now
    
    # state features
    for trajectory in trajectories:
        for i in range(len(trajectory)):
            # Compute the features         
            feature_dict = raw_trajectory_to_player_trajectory(trajectory[i], player_team, player_on_team)
            state = actor.select_features(agent.extractor, agent.get_feature_vector(**feature_dict))
            features.append( torch.as_tensor(state, dtype=torch.float32).view(-1) )

    it = 0

    for trajectory in trajectories:

        loss = []

        for i in range(len(trajectory)):                
            # Compute the returns

            action = DictObj(trajectory[i]['actions'][player_id])

            reward = actor.reward(
                action,
                agent.extractor,
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
    for it in range(iterations):
        batch_ids = torch.randint(0, len(returns), (batch_size,))
        batch_returns = returns[batch_ids]
        batch_actions = actions[batch_ids]
        batch_features = features[batch_ids]
        
        output = action_net(batch_features)
        pi = Bernoulli(probs=output)
        
        expected_log_return = (pi.log_prob(batch_actions).squeeze()*batch_returns).mean()
        optim.zero_grad()
        (-expected_log_return).backward()
        optim.step()
        avg_expected_log_return.append(float(expected_log_return))           
        

    action_net.eval()
        
    return context if context else {
        "score": None
    }

def validate_epoch( 
    team,
    agent,   
    actor,
    trajectories,
    epoch=0,
    evaluator=configuration.evaluator, 
    context=None):

    best_score = context["score"] if context is not None else None
    context = {
        "action_net":None,
        "score": None
    } if context is None else context

    # compute mean performance
    player_team = 0 # XXX
    player_on_team = 0 # XXX
    player_trajectories = [[raw_trajectory_to_player_trajectory(t, player_team, player_on_team) for t in trajectory] for trajectory in trajectories]
    score = evaluator.reduce(player_trajectories)

    print('epoch = %d dist = %s, best_dist = %s '%(epoch, score, best_score))
    
    if best_score is None or evaluator.is_better_than(score, best_score):
        context['action_net'] = copy.deepcopy(actor.action_net)
        context['score'] = score
        team.save()
        

    return context
                     

def train(
    team,
    epoch,  
    trajectories,  
    context=None
):
    return reinforce_epoch(team.agent, team.get_training_actor(), trajectories, epoch=epoch, context=context)

def validate(
    team,
    epoch,
    trajectories,
    context=None
):
    return validate_epoch(team, team.agent, team.get_training_actor(), trajectories, epoch=epoch, context=context)