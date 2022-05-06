# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

import copy
import torch
import numpy as np
from torch.distributions import Bernoulli, Normal
from state_agent.agents.subnets.rewards import SoccerBallDistanceObjective, OpponentDistanceObjective
from state_agent.agents.subnets.utils import DictObj, rollout_many
from state_agent.agents.subnets.agents import Agent
from state_agent.agents.subnets.features import OpponentFeatures

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

def get_player_action_tourney(t, player_id):
    return DictObj(t['actions'][player_id])


def raw_trajectory(t, player_team, player_on_team):
    return t

def get_player_action(t, player_id):
    return t['action']
class SoccerReinforcementConfiguration:

    mode = "soccer"

    def __init__(self, focus='puck') -> None:
        super().__init__()
        if focus == 'puck':
            self.evaluator = SoccerBallDistanceObjective()
        elif focus == 'opponent':
            self.evaluator = OpponentDistanceObjective()
        self.extract_trajectory = raw_trajectory
        self.extract_action = get_player_action
        self.agent = Agent      
        self.rollout_initializer = None  

class TournamentReinforcementConfiguration:

    mode = "soccer"

    def __init__(self) -> None:
        super().__init__()        
        self.evaluator = SoccerBallDistanceObjective() 
        self.extract_trajectory = raw_trajectory_to_player_trajectory
        self.extract_action = get_player_action_tourney
        self.agent = Agent       
        self.rollout_initializer = None


configuration = SoccerReinforcementConfiguration()

class Context():
    
    def __init__(self) -> None:
        self.score = None
        self.action_net = None
        self.actions=None
        self.rewards=None
        self.trajectories=None

def reinforce(
            actor, 
            actors,  
            configuration = configuration,              
            n_epochs = 10,
            n_trajectories = 100,
            n_iterations =100,
            n_validations = 20,
            n_steps=600,
            batch_size = 128,
            T = 20,
            epoch_post_process=None,
            focus='puck'
):
        
    slice_net = list(filter(lambda a: a != actor, actors))
    context = None

    for epoch in range(n_epochs):

        # perform the rollouts 
        actor.action_net.train()
        if focus == 'puck':
            agents = [configuration.agent(*slice_net, actor, train=True) for i in range(n_trajectories)]
            trajectories = rollout_many(agents, mode=configuration.mode, randomize=True, initializer=configuration.rollout_initializer, n_steps=n_steps)
        elif focus == 'opponent':
            agents = [
                configuration.agent(
                    *slice_net,
                    actor,
                    train=True,
                    extractor=OpponentFeatures()
                )
                for i in range(n_trajectories)
            ]
            trajectories = rollout_many(
                agents,
                mode=configuration.mode,
                randomize=True,
                initializer=configuration.rollout_initializer,
                n_steps=n_steps,
                num_karts=2,
                focus=focus,
                num_rollout=2
            )
        actor.action_net.eval()
        
        context = reinforce_epoch(
            agents, 
            actor, 
            trajectories,
            epoch=epoch,
            configuration=configuration,             
            batch_size=batch_size,
            iterations=n_iterations,
            context=context,
            focus=focus
        )

        # perform the validation rollouts
        if focus == 'puck':
            validation_agents = [configuration.agent(*slice_net, actor) for i in range(n_validations)]
            validation_trajectories = rollout_many(validation_agents, mode=configuration.mode, randomize=True, n_steps=n_steps)
        elif focus == 'opponent':
            validation_agents = [configuration.agent(*slice_net, actor, extractor=OpponentFeatures()) for i in range(n_validations)]
            validation_trajectories = rollout_many(
                validation_agents,
                mode=configuration.mode,
                randomize=True,
                n_steps=n_steps,
                num_karts=2,
                focus=focus,
                num_rollout=2
            )
        
        context, updated = validate_epoch(
            actor,
            validation_trajectories,
            configuration=configuration,
            epoch=epoch,            
            context=context,
            focus=focus
        )

        if epoch_post_process:
            epoch_post_process(actor, context)


    return context.action_net if context else None

def reinforce_epoch(
    agents,
    actor,    
    trajectories,  
    epoch=0,        
    configuration=configuration,
    iterations=100,      
    batch_size=128,
    reward_window=1,
    context=None,
    focus='puck'
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
    real_actions = []
    losses = []

    # XXX change these
    player_team = 0 # hard-coded for now
    player_on_team = 0 # hard-coded for now               
    player_id = 0 # hard-coded for now
    
    # state features
    last_kart_state = []
    last_action = None
    for num, trajectory in enumerate(trajectories):
        agent = agents[num % len(trajectories)] 
        for i in range(len(trajectory)):
            # Compute the features         
            feature_dict = configuration.extract_trajectory(trajectory[i], player_team, player_on_team)  # just returns trajectory[i]
            real_action = configuration.extract_action(trajectory[i], player_id)
            # state = actor.select_features(agent.extractor, agent.get_feature_vector(**feature_dict, last_state=last_kart_state, last_action=last_action))
            if focus == 'puck':  # TODO: check if last_state/last_action has an effect
                state = actor.select_features(agent.extractor, agent.get_feature_vector(**feature_dict, last_state=last_kart_state, last_action=last_action))
            elif focus == 'opponent':
                state = actor.select_features(agent.extractor, agent.get_feature_vector(feature_dict['kart_info'], feature_dict['kart_info_opp'], last_state=last_kart_state, last_action=last_action))
            features.append( torch.as_tensor(state, dtype=torch.float32).view(-1) )
            real_actions.append( real_action)
            last_kart_state.append( feature_dict['kart_info'] )
            if len(last_kart_state) > Agent.MAX_STATE:
                last_kart_state.pop(0)
            last_action = real_action

    it = 0

    action_net.train()
    for num, trajectory in enumerate(trajectories):

        loss = []
        agent = agents[num % len(trajectories)] 

        for i in range(len(trajectory)):                
            # Compute the returns
            features_vec = features[it + i]
            action = real_actions[it + i]
            greedy_action = actor.extract_greedy_action(action, features_vec)

            reward = actor.reward(
                action,
                greedy_action,
                features[it + i],
                features[it + min(i + T, len(trajectory)-1)]
            )
            
            loss.append(0)
        
            returns.append(reward) 
            
            # Store the action that we took
            actions.append( 
                greedy_action
            )

        it += len(trajectory)
        losses.append(np.sum(loss))
    
    # Upload everything to the GPU
    returns = torch.as_tensor(returns, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    features = torch.stack(features)

    print(returns)
    
    # enable this if not using discrete rewards! 
    #print(returns.mean(), returns.std())
    #returns = (returns - returns.mean()) / (returns.std() + 0.00001)
    avg_expected_log_return = []
    for it in range(iterations):
        batch_ids = torch.randint(0, len(returns), (batch_size,))
        batch_returns = returns[batch_ids]
        batch_actions = actions[batch_ids]
        batch_features = features[batch_ids]
        
        output = action_net(batch_features)
        log_prob = actor.log_prob(output, actions=batch_actions)

        expected_log_return = (log_prob.squeeze()*batch_returns).mean()
        optim.zero_grad()
        (-expected_log_return).backward()
        
        #actor.check_grad()
        optim.step()
        avg_expected_log_return.append(float(expected_log_return))           
        

    action_net.eval()

    context = context if context else Context()
    context.actions=actions.detach().numpy()
    context.rewards=returns.detach().numpy()
    context.trajectories=trajectories
        
    return context 

def validate_epoch(     
    actor,
    trajectories,
    epoch=0,
    configuration=configuration, 
    context=None):

    updated = False
    best_score = context.score if context is not None else None
    context = Context() if context is None else context

    # compute mean performance
    player_team = 0 # XXX
    player_on_team = 0 # XXX
    player_trajectories = [[configuration.extract_trajectory(t, player_team, player_on_team) for t in trajectory] for trajectory in trajectories]  # just returns the trajectory
    score = configuration.evaluator.reduce(player_trajectories)

    print('epoch = %d dist = %s, best_dist = %s '%(epoch, score, best_score))
    
    if best_score is None or configuration.evaluator.is_better_than(score, best_score):
        context.action_net = copy.deepcopy(actor.action_net)
        context.score = score
        updated = True
        
    return context, updated
                     

def train(
    team,
    epoch,  
    trajectories,  
    context=None
):
    return reinforce_epoch(team.agent, team.get_training_actor(), trajectories, epoch=epoch, context=context)[0]

def validate(
    team,
    epoch,
    trajectories,
    context=None
):
    context, updated = validate_epoch(team.get_training_actor(), trajectories, epoch=epoch, context=context)
    if updated:
        team.save()
