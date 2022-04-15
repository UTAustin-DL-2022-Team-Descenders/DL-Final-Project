from argparse import Action
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

# Assumes extract_features returns 31 channels
INPUT_CHANNELS = 31

# Assumes network outputs 6 different actions
# Output channel order: [acceleration, steer, brake, drift, fire, nitro]
# Boolean outputs will be thresholded by 0.5
OUTPUT_CHANNELS = 6

class ActionNetwork(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()

            self.network = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_output),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.network(x)

    def __init__(self, input_channels=INPUT_CHANNELS, hidden_layer_channels=[64, 128, 256], output_channels=OUTPUT_CHANNELS):
        super().__init__()

        layers = []
        for layer_channels in hidden_layer_channels:
            layers.append(ActionNetwork.Block(input_channels, layer_channels))
            input_channels = layer_channels

        layers.append(torch.nn.Linear(input_channels, output_channels))

        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Performs Reinforcement training for an ActionNetwork.
# Using Deep Q Learning
class ActionNetworkTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9, optimizer="ADAM"):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # Select optimizer
        if optimizer.upper() == "ADAM":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        
        self.loss_module = torch.nn.MSELoss()

    def train_step(self, prev_state_features, action, reward, curr_state_features, done):

        # Skip training for very first time step
        if prev_state_features is None:
            return

        # Convert inputs to Tensors
        if isinstance(prev_state_features, tuple):
            prev_state_features = torch.stack(prev_state_features)
            curr_state_features = torch.stack(curr_state_features)
            action = torch.stack(action)
            reward = torch.stack(reward)
        else:
            prev_state_features = torch.unsqueeze(prev_state_features, 0)
            curr_state_features = torch.unsqueeze(curr_state_features, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        ## Deep Learning Q START
        # 1: predicted Q values with current prev_state_features
        pred = self.model(prev_state_features)
        
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        target = pred.clone()
        
        # Iterate over items in batch
        for idx in range(len(done)):
        
            # default Q_new is simply the reward
            Q_new = reward[idx]
        
            # If not done, Q_new is reward + discount_rate_gamma * <highest confidence action for current state>
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * self.model(curr_state_features[idx])
        
            # Set the target's action to Q_new
            target[idx] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.loss_module(target, pred)
        loss.backward()
        # Deep Learning Q END

        # Reinforce START
        # forward feed features through action network and get output
        #output = self.model(prev_state_features)
        #
        ## Create a steering policy from Bernoulli distribtuion of network output
        #steer_policy = Bernoulli(probs=output[:,1])
        #
        ## expected log return is the log probability of the policy for the actions taken (batch_actions) times the returns we gotten.
        ## Then take the mean
        #expected_log_return = (steer_policy.log_prob(action[:,1] > 0)*reward).mean()
        #self.optimizer.zero_grad()
        #
        ## Take the negative expected log return to call backwards
        #(-expected_log_return).backward()
        # Reinforce END

        self.optimizer.step()


# StateAgent agnostic Save & Load model functions. Used in state_agent.py Match
def save_model(model):
    from os import path
    model_scripted = torch.jit.script(model)
    model_scripted.save(path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt' ))


def load_model():
    from os import path
    import sys
    load_path = path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt')
    try:
        model = torch.jit.load(load_path)
        print("Loaded pre-existing ActionNetwork from", load_path)
        return model
    except FileNotFoundError as e:
        print("Couldn't find existing model in %s" % load_path)
        sys.exit()
    except ValueError as e:
        print("Couldn't find existing model in %s" % load_path)
        sys.exit()