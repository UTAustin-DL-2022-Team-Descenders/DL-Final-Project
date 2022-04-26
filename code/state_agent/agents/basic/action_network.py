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

HIDDEN_LAYER_1_SIZE = 400
HIDDEN_LAYER_2_SIZE = 300

class ActionNetwork(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()

            self.network = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_output),
                torch.nn.BatchNorm1d(n_output),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.network(x)

    def __init__(self, input_channels=INPUT_CHANNELS, hidden_layer_channels=[HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE], output_channels=OUTPUT_CHANNELS):
        super().__init__()

        layers = []
        for layer_channels in hidden_layer_channels:
            layers.append(ActionNetwork.Block(input_channels, layer_channels))
            input_channels = layer_channels

        self.main_network = torch.nn.Sequential(*layers)

        self.acceleration_net = torch.nn.Sequential(torch.nn.Linear(input_channels, 1),
                                                    torch.nn.Sigmoid())

        self.steering_net = torch.nn.Sequential(torch.nn.Linear(input_channels, 1),
                                                    torch.nn.Tanh())

        self.brake_net = torch.nn.Sequential(torch.nn.Linear(input_channels, 1),
                                                    torch.nn.Sigmoid())

        self.drift_net = torch.nn.Sequential(torch.nn.Linear(input_channels, 1),
                                                    torch.nn.Sigmoid())

        self.fire_net = torch.nn.Sequential(torch.nn.Linear(input_channels, 1),
                                                    torch.nn.Sigmoid())

        self.nitro_net = torch.nn.Sequential(torch.nn.Linear(input_channels, 1),
                                                    torch.nn.Sigmoid())

        #layers.append(torch.nn.Linear(input_channels, output_channels))
        #self.network = torch.nn.Sequential(*layers)

        
    def forward(self, x):

        x = self.main_network(x)
        
        acceleration = self.acceleration_net(x)
        steering = self.steering_net(x)
        brake = self.brake_net(x)
        drift = self.drift_net(x)
        fire = self.fire_net(x)
        nitro = self.nitro_net(x)

        output = torch.cat([acceleration, steering, brake, drift, fire, nitro], dim=-1)

        return output

class CriticNetwork(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()

            self.network = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_output),
                torch.nn.BatchNorm1d(n_output),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.network(x)

    def __init__(self, input_state_channels=INPUT_CHANNELS, input_action_channels=OUTPUT_CHANNELS):
        super(CriticNetwork, self).__init__()

        layers = []

        self.layer1 = CriticNetwork.Block(input_state_channels + input_action_channels, HIDDEN_LAYER_1_SIZE)
        self.layer2 = CriticNetwork.Block(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE)
        self.layer3 = torch.nn.Linear(HIDDEN_LAYER_2_SIZE, 1)

        self.network = torch.nn.Sequential(*layers)

    def forward(self, states, actions):

        states_actions = torch.cat([states, actions], 1)
        out = self.layer1(states_actions)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

# Performs Reinforcement training for an ActionNetwork.
# Using Deep Q Learning
class ActionCriticNetworkTrainer:
    def __init__(self, action_net, target_action_net, critic_net, target_critic_net, 
                    lr=0.001, discount_rate=0.9, optimizer="ADAM", tau=0.001,
                    logger=None):
        self.lr = lr
        self.discount_rate = discount_rate
        self.action_net = action_net
        self.target_action_net = target_action_net
        self.critic_net = critic_net
        self.target_critic_net = target_critic_net
        self.tau = tau
        self.logger = logger

        # Select optimizer
        if optimizer.upper() == "ADAM":
            self.action_optimizer = torch.optim.Adam(action_net.parameters(), lr=self.lr)
            self.critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=self.lr)
        else:
            self.action_optimizer = torch.optim.SGD(action_net.parameters(), lr=self.lr, momentum=0.9)
            self.critic_optimizer = torch.optim.SGD(critic_net.parameters(), lr=self.lr, momentum=0.9)
        
        self.loss_module = torch.nn.MSELoss()

    # Perform a training step using a Reinforcement algorithm
    def train_step(self, curr_state_features, action, reward, next_state_features, not_done, global_step):

        # Convert inputs to Tensors
        if isinstance(curr_state_features, tuple):
            curr_state_features = torch.stack(curr_state_features)
            next_state_features = torch.stack(next_state_features)
            action = torch.cat(action, 0)
            reward = torch.unsqueeze(torch.stack(reward), 1)
            not_done = torch.unsqueeze(torch.stack(not_done), 1)

        else:
            curr_state_features = torch.unsqueeze(curr_state_features, 0)
            next_state_features = torch.unsqueeze(next_state_features, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            not_done = torch.unsqueeze(not_done, 0)

        self.action_net.train()
        self.critic_net.train()

        # Normalize rewards
        #reward = (reward - reward.mean()) / (reward.std() + 0.0001)

        ## Deep Deterministic Policy Gradient START

        # Critic Loss
        q_values = self.critic_net(curr_state_features, action)

        pred_actions = self.target_action_net(next_state_features)
        next_q = self.target_critic_net(next_state_features, pred_actions)

        target_q = reward + (not_done * self.discount_rate * next_q).detach()

        critic_loss = self.loss_module(q_values, target_q)

        if self.logger:
            self.logger.add_scalar("critic_loss", critic_loss, global_step=global_step)
        
        # Critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.critic_net.eval()

        # Action network Loss
        policy_loss = -self.critic_net(curr_state_features, self.action_net(curr_state_features)).mean()

        if self.logger:
            self.logger.add_scalar("policy_loss", policy_loss, global_step=global_step)
        
        # Action network update
        self.action_optimizer.zero_grad()
        policy_loss.backward()
        self.action_optimizer.step()

        # Perform soft network update from action/critic network to target networks
        self.soft_network_update(self.action_net, self.target_action_net, self.tau)
        self.soft_network_update(self.critic_net, self.target_critic_net, self.tau)
        ## Deep Deterministic Policy Gradient END

        # REINFORCE START
        # forward feed features through action network and get output
        #output = self.model(curr_state_features)
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
        # REINFORCE END

        #self.optimizer.step()
    
    # Perform a soft network update of target network from source network using scaling factor tau
    def soft_network_update(self, src_net, target_net, tau):
        for target_param, param in zip(target_net.parameters(), src_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


# StateAgent agnostic Save & Load model functions
def save_model(model, name="state_agent"):
    from os import path
    model.eval()
    model_scripted = torch.jit.script(model)
    model_scripted.save(path.join(path.dirname(path.abspath(__file__)), '%s.pt' % name))


def load_model(name="state_agent"):
    from os import path
    import sys
    load_path = path.join(path.dirname(path.abspath(__file__)), '%s.pt' % name)
    try:
        model = torch.jit.load(load_path)
        print("Loaded pre-existing model from", load_path)
        return model
    except FileNotFoundError as e:
        print("Couldn't find existing model in %s" % load_path)
        sys.exit()
    except ValueError as e:
        print("Couldn't find existing model in %s" % load_path)
        sys.exit()