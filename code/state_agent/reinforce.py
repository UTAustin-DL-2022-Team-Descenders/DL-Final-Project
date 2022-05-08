from argparse import Namespace
import torch
import copy
from torch.distributions import Bernoulli
from state_agent.utils import get_pickle_files

BATCH_SIZE = 128
COMPARISON_ACTOR = None
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def decode_action(action):
    return [
            action['acceleration'].item() if hasattr(action, 'acceleration') else 0.0,
            action['steer'].item() if hasattr(action, 'steer') else 0.0,
            action['brake'].item() if hasattr(action, 'brake') else 0.0,
            action['drift'].item() if hasattr(action, 'drift') else 0.0,
            action['fire'].item() if hasattr(action, 'fire') else 0.0,
            action['nitro'].item() if hasattr(action, 'nitro') else 0.0
    ]

def find_best_actor():
    pass

def evaluate_trajectory(actor, trajectory, step):    
    return  actor.evaluate_trajectory(step, trajectory)

def extract_features(actor, trajectory):
    return  actor.extract_features(trajectory)

def extract_actions(actor, trajectory):
    # XXX Assume agent is first player    
    return list(decode_action(trajectory['actions'][0]))

def rollout_actors(rollout, actors, comparison_actor):
    return [rollout(
        Namespace(team1=actor, team2=comparison_actor) 
    ) for actor in actors]

def validate_and_update_agent(rollout, best_actor, actor, comparison_actor):
    # validate 
    (_, best_performance), (_, current_performance) = rollout_actors(rollout, [best_actor, actor], comparison_actor)     
    if evaluate_trajectory(best_actor, best_performance[0], -1) < evaluate_trajectory(actor, current_performance[0], -1):
        # save the new best actor
        return actor.save()
    
def train(args):

    actor = args.actor
    comparison_actor = args.comparison_actor
    action_net = args.actor.model
    iterations = args.iterations
    trajectories = args.trajectories
    rollout = args.rollout

    if type(trajectories) == str:
        # load the trajectories
        trajectories = get_pickle_files(trajectories)
    
    best_action_net = copy.deepcopy(action_net)

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)
    
    # Compute all the reqired quantities to update the policy
    features = []
    returns = []
    actions = []
    for trajectory in trajectories:
        for i in range(len(trajectory)):
            # Compute the returns
            returns.append( 
                evaluate_trajectory(actor, trajectory, i)
            )
            # Compute the features
            features.append( torch.as_tensor(extract_features(actor, trajectory[i]), dtype=torch.float32).view(-1) )
            # Store the actions that we took
            actions_dec= extract_actions(actor, trajectory[i])            
            actions.append(actions_dec)
    
    # Upload everything to the GPU
    returns = torch.as_tensor(returns, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    features = torch.stack(features)    
    returns = (returns - returns.mean()) / returns.std()
    
    # train
    action_net.train()
    avg_expected_log_return = []
    for it in range(iterations):
        batch_ids = torch.randint(0, len(returns), (BATCH_SIZE,), device=DEVICE)
        batch_returns = returns[batch_ids]        
        batch_actions = actions[batch_ids]
        batch_features = features[batch_ids]
        
        output = action_net(batch_features)
        pi = Bernoulli(logits=output)
        log_prob = pi.log_prob(batch_actions)        
        batch_returns = batch_returns.expand(batch_actions.shape[1], -1).permute(1, 0)
        expected_log_return = (batch_returns*log_prob).mean()
        optim.zero_grad()
        (-expected_log_return).backward()
        optim.step()
        avg_expected_log_return.append(float(expected_log_return))

    # validation
    best_action_net = validate_and_update_agent(rollout, actor.__class__(best_action_net), actor.__class__(action_net), comparison_actor)

def main(args_local=None):

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--iterations', default=100, type=int, help="Number of gradient descent steps per epoch")
    parser.add_argument('--trajectories', type=str, help="Path to trajectories folder")
    
    args = parser.parse_known_args()[0]

    # merge args
    if args:
        new_dict: dict = vars(args).copy()   # start with keys and values of starting_dict
        new_dict.update(vars(args_local))
        args = argparse.Namespace(**new_dict)
    
    train(args)

    