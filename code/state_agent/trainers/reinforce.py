BATCH_SIZE = 128

def find_best_actor():
    pass

def evaluate_trajectories(actor, trajectories, step):
    return  actor.evaluate_trajectories(step, trajectories)

def extract_features(actor, trajectory):
    return  actor.extract_features(trajectory)

def main(args):

    import copy

    action_net = args.action_net
    n_epochs = args.epochs
    trajectories = args.trajectories
    
    T = 20
    
    best_action_net = copy.deepcopy(action_net)

    optim = torch.optim.Adam(action_net.parameters(), lr=1e-3)


    for epoch in range(n_epochs):
        eps = 1e-2
        
        # Compute all the reqired quantities to update the policy
        features = []
        returns = []
        actions = []
        for trajectory in trajectories:
            for i in range(len(trajectory)):
                # Compute the returns
                returns.append( 
                    evaluate_trajectories(action_net, trajectories, i)
                )
                # Compute the features
                features.append( torch.as_tensor(extract_features(trajectory[i]), dtype=torch.float32).cuda().view(-1) )
                # Store the action that we took
                actions.append( trajectory[i]['action'].steer > 0 )
        
        # Upload everything to the GPU
        returns = torch.as_tensor(returns, dtype=torch.float32).cuda()
        actions = torch.as_tensor(actions, dtype=torch.float32).cuda()
        features = torch.stack(features).cuda()
        
        returns = (returns - returns.mean()) / returns.std()
        
        action_net.train().cuda()
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
        if best_performance[-1]['kart_info'].overall_distance < current_performance[-1]['kart_info'].overall_distance:
            best_action_net = copy.deepcopy(action_net)