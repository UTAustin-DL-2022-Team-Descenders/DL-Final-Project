from os import path
from turtle import pu 
import numpy as np
from state_agent.agents.basic.action_network import load_model, save_model
import torch
import collections
from .action_network import ActionNetwork, CriticNetwork, ActionCriticNetworkTrainer, save_model, load_model
import random, copy
import torch.nn.functional as F

# Globals
DEBUG_EN = False
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MAX_SCORE = 3
GOAL_LINE_Y_BUFFER = 3

# Hyperparameters
OBJS_TOUCHING_DISTANCE_THRESHOLD = 4 # Pixel distance threshold to denote objects are touching
ACTION_TENSOR_EPSILON_THRESHOLD = 100 # used in get_random_action for decaying exploration vs exploitation

# StateAgent Default values
DISCOUNT_RATE_GAMMA = 0.9 # between 0 to 1
STATE_CHANNELS = 31 # State Channels created by get_features
ACTION_CHANNELS = 6 # One channel for each actions

class StateAgent:

  MAX_MEMORY = 1000000 # Number of timesteps to memorize
  BATCH_SIZE = 128 # Batch size for long term training

  def __init__(self, input_channels=STATE_CHANNELS, output_channels=ACTION_CHANNELS, 
                discount_rate=DISCOUNT_RATE_GAMMA, load_existing_model=False, optimizer="ADAM", lr=0.001,
                logger=None, player_num=0, noise_std=0.01):

    # number of games played
    self.n_games = 0 

    # Player num of this agent, used to differentiate using multiple agents on a team
    self.player_num = player_num

    # Collect entries for trained timesteps
    self.memory = collections.deque(maxlen=StateAgent.MAX_MEMORY)

    # Noise to add to ActionNetwork output 
    self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std) * np.ones(1))

    # Try loading an existing action_net to continue training it
    if load_existing_model:
      self.action_net = load_model("state_agent")
      self.critic_net = load_model("critic")
    else:
      # Initialize an ActionNetwork
      self.action_net = ActionNetwork(input_channels=input_channels, 
                              output_channels=output_channels)

      # Save this newly initialized action_net for rollouts
      self.action_net.to(DEVICE)
      save_model(self.action_net)

      self.critic_net = CriticNetwork(input_state_channels=STATE_CHANNELS, input_action_channels=ACTION_CHANNELS)
      self.critic_net.to(DEVICE)

    self.target_action_net = ActionNetwork(input_channels=input_channels, output_channels=output_channels)
    self.target_action_net.to(DEVICE)
    self.target_critic_net = CriticNetwork(input_state_channels=STATE_CHANNELS, input_action_channels=ACTION_CHANNELS)
    self.target_critic_net.to(DEVICE)

    # Make target networks copies of original action/critic networks
    self.copy_network_params(self.action_net, self.target_action_net)
    self.copy_network_params(self.critic_net, self.target_critic_net)

    # Initialize a trainer to perform network training
    self.trainer = ActionCriticNetworkTrainer(action_net=self.action_net, target_action_net=self.target_action_net,
                                              critic_net=self.critic_net, target_critic_net=self.target_critic_net,
                                              discount_rate=discount_rate, lr=lr, optimizer=optimizer,
                                              logger=logger)
  
  def copy_network_params(self, source_network, destination_network):

    for dest_param, src_param in zip(destination_network.parameters(), source_network.parameters()):
      dest_param.data.copy_(src_param.data)

  # Add timestep information to memory
  def memorize(self, curr_state, action, reward, next_state, not_done):

    if not_done == True:
      not_done = torch.tensor(1, device=DEVICE)
    else:
      not_done = torch.tensor(0, device=DEVICE)

    # Save information as a tuple
    self.memory.append((curr_state, action, reward, next_state, not_done))


  # Performs training step on a batch of timesteps from memory
  def train_batch_timesteps(self, global_step):

    # Get a sample if memory contains at least a batch size
    if len(self.memory) > StateAgent.BATCH_SIZE:
      mini_sample = random.sample(self.memory, StateAgent.BATCH_SIZE)
    else:
      mini_sample = self.memory # Otherwise just take the entire memory as a batch

    # Convert mini_sample into a list of state_features, actions, rewards, & dones
    curr_states_features, actions, rewards, next_states_features, dones = zip(*mini_sample)

    self.trainer.train_step(curr_states_features, actions, rewards, next_states_features, dones, global_step)


  # Performs training step on a single timestep
  def train_immediate_timestep(self, curr_state_features, action, reward, next_state_features, not_done, global_step):
    self.trainer.train_step(curr_state_features, action, reward, next_state_features, not_done, global_step)


  # Gets an action tensor for a timestep
  def get_action_tensor(self, state_features):
    
    epsilon = ACTION_TENSOR_EPSILON_THRESHOLD - self.n_games
    actions_min_bounds = torch.tensor((0, -1, 0, 0, 0, 0), device=DEVICE) # Minimum values of each action
    actions_max_bounds = torch.tensor((1, 1, 1, 1, 1, 1), device=DEVICE) # maximum values of each action

    # Either take a random action (exploration)
    # or prediction action from the action_net (exploitation).
    # Likeihood of random action decays as number of games increases

    self.action_net.eval()
    state_features = torch.unsqueeze(state_features, 0)
    action_tensor = self.action_net(state_features).detach()

    action_tensor = torch.squeeze(action_tensor, 0)

    #noise = self.noise()
    #action_tensor = action_tensor + torch.as_tensor(noise, dtype=torch.float32, device=DEVICE)
    #action_tensor = torch.clamp(action_tensor, actions_min_bounds, actions_max_bounds)

    #if random.randint(0, 150) < epsilon:
      
      #action_tensor[0] = self.noise.sample(action_tensor[0])
      #actions_min_bounds = torch.tensor((0, -1, 0, 0, 0, 0), device=DEVICE) # Minimum values of each action
      #actions_max_bounds = torch.tensor((1, 1, 1, 1, 1, 1), device=DEVICE) # maximum values of each action
      #action_tensor[0] += torch.tensor(self.ou_noise.sample(), device=DEVICE)
      #action_tensor[0] = torch.clamp(action_tensor[0], actions_min_bounds, actions_max_bounds)

    print("get_action_tensor - ", action_tensor)
    return action_tensor
  

  # Use thresholding for boolean actions
  def get_random_action(self):
    acceleration = random.random() # between 0 to 1
    steer = random.uniform(-1, 1)
    brake = random.random() > 0.5
    drift = random.random() > 0.5
    fire = random.random() > 0.5
    nitro = random.random() > 0.5

    action_tensor = torch.tensor((acceleration,
                        steer,
                        brake,
                        drift,
                        fire,
                        nitro),
                      device=DEVICE)
    return torch.unsqueeze(action_tensor, 0)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
class NormalNoise:

  def __init__(self, size, mean=[0.5, 0, 0.5, 0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], action_mins=[0, -1, 0, 0, 0, 0], action_maxs=[1, 1, 1, 1, 1, 1]):

    self.means = torch.tensor(mean, device=DEVICE)
    self.stds = torch.tensor(std, device=DEVICE)
    self.action_mins = torch.tensor(action_mins, device=DEVICE)
    self.action_maxs = torch.tensor(action_maxs, device=DEVICE)

  def sample(self, action):
    action += torch.normal(self.means, self.stds)
    return torch.clamp(action, self.action_mins, self.action_maxs)
class Team:
    agent_type = 'state'
    
    def __init__(self, model=None):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        from .action_network import ActionNetwork
        self.team = None
        self.num_players = None
        self.training_mode = None
        if model is None:
          try:
            self.model = load_model()
          except:
            self.model = ActionNetwork()
        else:
          self.model = model
    
    def save(self):
        save_model(self.model)

    def set_training_mode(self, mode):
        """
        The training mode algorithm will be passed by name here.
        This allows the agent to decide what actions to take based on the training type.
        For example, a "reinforce" mode could use randomized actions from a policy distribution 
        """
        self.training_mode = mode

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_states, opponent_states, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_states: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_states: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # actions list for all players on this team
        actions = []

        # Iterate over all players on this team
        for player_num, player_state in enumerate(player_states):

          # Get features from this player, teammates, opponents, and ball
          player_features = get_features(player_state, player_states[:player_num] + player_states[player_num+1:], opponent_states, soccer_state, self.team)

          player_features = torch.unsqueeze(player_features, 0)

          # Get network output by forward feeding features through the model
          network_output = self.model(player_features)[0]

          # add action dictionary to actions list
          actions.append(get_action_dictionary_from_network_output(network_output))

        return actions

    def evaluate_trajectory(self, step, trajectory):
        # how close to the puck?    
        if step == -1:
          step = len(trajectory) - 1         
        puck_locations = [get_puck_center(t['soccer_state']) for t in trajectory[step:min(step+20,len(trajectory))]]
        puck_locations = torch.cat(puck_locations)
        kart_locations = [get_kart_center(t['team_state'][self.team][0]) for t in trajectory[step:min(step+20,len(trajectory))]]
        kart_locations = torch.cat(kart_locations)      
        return torch.nn.functional.mse_loss(kart_locations, puck_locations)        

    def extract_features(self, state):
        # XXX extract the first players actions??       
        player_states, opponent_states, soccer_state = state['team_state'][0], state['team_state'][1], state['soccer_state']
        player_features = get_features(player_states[0], player_states[:0] + player_states[1:], opponent_states, soccer_state, self.team)
        return player_features


def get_features_from_unified_state_dictionaries(state_dictionaries, team_num, player_num):

  # Get agent & opponent dictionary key info
  player_team_key = "team%0d_state" % team_num
  opponent_team_num = 2 if team_num == 1 else 1
  opponent_team_key = "team%0d_state" % opponent_team_num

  # Get player & opponent state dictionaries
  player_states = state_dictionaries[player_team_key]
  opponent_states = state_dictionaries[opponent_team_key]
  
  return get_features(player_states[player_num], player_states[:player_num] + player_states[player_num+1:], 
                                                opponent_states, state_dictionaries["soccer_state"],
                                                team_num)

def get_features(player_state, team_state, opponent_states, puck_state, team_num):

    opponent_team_num = 2 if team_num == 1 else 1

    # Collect features inside a dictionary
    features_dict = {}

    # Get player features
    features_dict["player_kart_front"] = get_kart_front(player_state)
    features_dict["player_kart_center"] = get_kart_center(player_state)
    features_dict["player_kart_angle"] = get_obj1_to_obj2_angle(features_dict["player_kart_front"], features_dict["player_kart_center"])

    # Get puck features
    features_dict["puck_center"] = get_puck_center(puck_state)
    features_dict["kart_to_puck_angle"]  = get_obj1_to_obj2_angle(features_dict["player_kart_center"], features_dict["puck_center"]) 

    # Get goal line features
    # - Opponent goal line
    features_dict["opponent_goal_line_center"] = get_team_goal_line_center(puck_state, get_goal_line_index_from_team_num(opponent_team_num))
    features_dict["puck_to_opponent_goal_line_angle"] = get_obj1_to_obj2_angle(features_dict["puck_center"], features_dict["opponent_goal_line_center"])
    features_dict["kart_to_opponent_goal_line_difference"] = get_obj1_to_obj2_angle_difference(features_dict["player_kart_angle"], features_dict["puck_to_opponent_goal_line_angle"])

    # - Team goal line
    features_dict["team_goal_line_center"] = get_team_goal_line_center(puck_state, get_goal_line_index_from_team_num(team_num))
    features_dict["puck_to_team_goal_line_angle"] = get_obj1_to_obj2_angle(features_dict["puck_center"], features_dict["team_goal_line_center"])
    features_dict["kart_to_team_goal_line_difference"] = get_obj1_to_obj2_angle_difference(features_dict["player_kart_angle"], features_dict["puck_to_team_goal_line_angle"])

    # Get opponent features
    for opponent_id, opponent_state in enumerate(opponent_states):
      opponent_name = "opponent_%0d" % opponent_id
      features_dict["%s_center" % opponent_name] = opponent_center = get_kart_center(opponent_state)
      features_dict["player_to_%s_angle" % opponent_name] = player_to_opponent_angle = get_obj1_to_obj2_angle(features_dict["player_kart_center"], opponent_center)
      features_dict["player_to_%s_angle_difference" % opponent_name] = get_obj1_to_obj2_angle_difference(features_dict["player_kart_center"], player_to_opponent_angle)

    # Get teammate features
    for teammate_id, teammate_state in enumerate(team_state):
      teammate_name = "teammate_%0d" % teammate_id
      features_dict["%s_center" % teammate_name] = teammate_center = get_kart_center(teammate_state)
      features_dict["player_to_%s_angle" % teammate_name] = player_to_teammate_angle = get_obj1_to_obj2_angle(features_dict["player_kart_center"], teammate_center)
      features_dict["player_to_%s_angle_difference" % teammate_name] = get_obj1_to_obj2_angle_difference(features_dict["player_kart_center"], player_to_teammate_angle)

    # Flatten out features_dictionary into a feature_list
    features_list = []
    for value in features_dict.values():

      # Multi-dimensional value tensors need to be unrolled
      if value.size() and value.size()[0] > 1:
        for x in value:
          features_list.append(x)
          
      else:
        features_list.append(value)

    if DEBUG_EN:
      print("printing feature_list dictionary")
      for key, value in features_dict.items():
        print(key, " - ", value)
      
      print("printing features_list")
      for feature in features_list:
        print(feature)

    return torch.tensor(features_list, dtype=torch.float32, device=DEVICE)

def get_kart_front(state):
  return torch.tensor(state['kart']['front'], dtype=torch.float32)[[0, 2]]

def get_kart_center(kart_state):
  return get_object_center(kart_state["kart"])
  
def get_obj1_to_obj2_angle(object1_center, object2_center):
  object1_direction = get_obj1_to_obj2_direction(object1_center, object2_center)
  return torch.atan2(object1_direction[1], object1_direction[0])

def get_obj1_to_obj2_direction(object1_center, object2_center):
  return (object2_center-object1_center) / torch.norm(object2_center-object1_center)

def get_object_center(state_dict):
  return torch.tensor(state_dict['location'], dtype=torch.float32)[[0, 2]]

def get_puck_center(puck_state):
  return get_object_center(puck_state["ball"])

def get_team_goal_line_center(puck_state, goal_line_index):
  return torch.tensor(puck_state['goal_line'][goal_line_index], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

def get_team_goal_line(puck_state, goal_line_index):
  return torch.tensor(puck_state['goal_line'][goal_line_index], dtype=torch.float32)[:, [0, 2]]

# limit angle between -1 to 1
def limit_period(angle):
  return angle - torch.floor(angle / 2 + 0.5) * 2 

def get_obj1_to_obj2_angle_difference(object1_angle, object2_angle):
  angle_difference = (object1_angle - object2_angle) / np.pi
  return limit_period(angle_difference)

# Assumes there are always two teams (0 and 1)
# Team 1 -> Goal 0; Team 2 -> Goal 1
def get_goal_line_index_from_team_num(team_num):
  if team_num == 1:
    return 0
  else:
    return 1

# Assumes network outputs 6 different actions
# Output channel order: [acceleration, steer, brake, drift, fire, nitro]
# REVIST: Using simple thresholding for boolean actions for now
def get_action_dictionary_from_network_output(network_output):
    return dict(acceleration=network_output[0],
                steer=network_output[1],
                brake=network_output[2] > 0.5,
                drift=network_output[3] > 0.5,
                fire=network_output[4]  > 0.5,
                nitro=network_output[5] > 0.5
                )


# TODO: Make this reward function more robust
def get_reward(state_dictionaries, actions, team_num, player_num):

  DEBUG_EN = True
  reward = 0

  # Get agent & opponent dictionary key info
  player_team_key = "team%0d_state" % team_num
  opponent_team_num = 2 if team_num == 1 else 1
  opponent_team_key = "team%0d_state" % opponent_team_num

  # Get player, team, and opponent state dictionaries
  player_and_team_states = state_dictionaries[player_team_key]
  player_state = player_and_team_states[player_num]
  team_state = player_and_team_states[:player_num] + player_and_team_states[player_num+1:]
  opponent_states = state_dictionaries[opponent_team_key]

  # Get puck_state
  puck_state = state_dictionaries["soccer_state"]

  player_goal_line_index = get_goal_line_index_from_team_num(team_num)
  opponent_goal_line_index = get_goal_line_index_from_team_num(opponent_team_num)

  opponent_goal_center = get_team_goal_line_center(puck_state, opponent_goal_line_index)
  player_goal_center = get_team_goal_line_center(puck_state, player_goal_line_index)

  puck_center = get_puck_center(puck_state)
  kart_center = get_kart_center(player_state)

  kart_to_puck_distance = get_obj1_to_obj2_distance(kart_center, puck_center)
  puck_to_opponent_goal_distance = get_obj1_to_obj2_distance(opponent_goal_center, puck_center)
  puck_to_player_goal_distance = get_obj1_to_obj2_distance(player_goal_center, puck_center)
  
  #if DEBUG_EN:
    #print("get_reward - kart_to_puck_distance ", kart_to_puck_distance)
    #print("get_reward - inverse kart_to_puck_distance ", 10*(1/kart_to_puck_distance))
    #print("get_reward - puck_to_opponent_goal_distance ", puck_to_opponent_goal_distance)
    #print("get_reward - inverse puck_to_opponent_goal_distance ", 10*(1/puck_to_opponent_goal_distance))
    #print("get_reward - inverse puck_to_player_goal_distance ", 10*(1/puck_to_player_goal_distance))


  # Increase reward as puck gets closer to the opponent goal
  reward += (1/puck_to_opponent_goal_distance)*10

  # Decrease reward as puck get closer to player goal
  reward -= (1/puck_to_player_goal_distance)*10

  # Increase reward as kart gets closer to the puck
  reward += (1/kart_to_puck_distance)*10

  if is_touching(kart_center, puck_center):
    reward += 10
    if DEBUG_EN:
      print("player is touching the puck")

  if is_puck_in_team_goal(puck_state, team_num):
    reward -= 100
    if DEBUG_EN:
      print("puck is in agent goal")

  if is_puck_in_team_goal(puck_state, opponent_team_num):
    reward += 100
    if DEBUG_EN:
      print("puck is in opponent goal")
  
  #return torch.tensor(reward, dtype=torch.float32, device=DEVICE)
  return reward.clone().detach().to(DEVICE)

def is_touching(object1_center, object2_center, threshold=OBJS_TOUCHING_DISTANCE_THRESHOLD):
  return get_obj1_to_obj2_distance(object1_center, object2_center) < threshold

def is_puck_in_team_goal(puck_state, team_num):

  puck_center = get_puck_center(puck_state)
  team_goal = get_goal_line_index_from_team_num(team_num)
  team_goal_line = get_team_goal_line(puck_state, team_goal)

  # The puck is in goal if the puck_center is in between the two sides of the goal
  # and past the Y axis of the goal line minus a small buffer

  if team_num == 1:

    # puck is in between goal posts
    if team_goal_line[0][0] <= puck_center[0] <= team_goal_line[1][0]:

      # if puck is at or above team1 goal line (-64.5 + buffer)
      if puck_center[1] <= (team_goal_line[0][1] + GOAL_LINE_Y_BUFFER):
        return True
  
  else: # team_num == 2
    
    # puck is in between goal posts
    if team_goal_line[1][0] <= puck_center[0] <= team_goal_line[0][0]:

      # if puck is at or below team2 goal line (64.5 - buffer)
      if puck_center[1] >= (team_goal_line[0][1] - GOAL_LINE_Y_BUFFER):
          return True
      
  return False

def get_obj1_to_obj2_distance(object1_center, object2_center):
  return F.pairwise_distance(object1_center, object2_center)

# Get the score of a team using team_num
def get_score(puck_state, team_num):
  return puck_state["score"][get_goal_line_index_from_team_num(team_num)]
