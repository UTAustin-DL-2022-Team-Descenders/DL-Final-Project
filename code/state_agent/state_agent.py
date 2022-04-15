from os import path 
import numpy as np
import torch
import collections
from .action_network import ActionNetwork, ActionNetworkTrainer, save_model, load_model
import random
import torch.nn.functional as F

# Globals
DEBUG_EN = False
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MAX_SCORE = 3
GOAL_LINE_Y_BUFFER = 1

# Hyperparameters
OBJS_TOUCHING_DISTANCE_THRESHOLD = 4 # Pixel distance threshold to denote objects are touching
ACTION_TENSOR_EPSILON_THRESHOLD = 9 # used in get_random_action for decaying exploration vs exploitation

# StateAgent Default values
DISCOUNT_RATE_GAMMA = 0.9 # between 0 to 1
INPUT_CHANNELS = 31 # Network input channels created by get_features
OUTPUT_CHANNELS = 6 # Network outputs used for actions

class StateAgent:

  MAX_MEMORY = 100000 # Number of timesteps to memorize
  BATCH_SIZE = 16 # Batch size for long term training

  def __init__(self, input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS, 
                gamma=DISCOUNT_RATE_GAMMA, load_existing_model=False, optimizer="ADAM", lr=0.001):
    self.n_games = 0 # number of games played

    # Collect entries for trained timesteps
    self.memory = collections.deque(maxlen=StateAgent.MAX_MEMORY)

    # Try loading an existing model to continue training it
    if load_existing_model:
      self.model = load_model()
    else:
      # Initialize an ActionNetwork
      self.model = ActionNetwork(input_channels=input_channels, 
                              output_channels=output_channels)

      # Save this newly initialized model for rollouts
      self.model.to(DEVICE)
      save_model(self.model)

    # Initialize a trainer to perform network training
    self.trainer = ActionNetworkTrainer(self.model, gamma=gamma, lr=lr, optimizer=optimizer)
  

  # Add timestep information to memory
  def memorize(self, prev_state, action, reward, curr_state, done):

    # Return if prev_state is None
    if prev_state is None:
      return

    # Save information as a tuple
    self.memory.append((prev_state, action, reward, curr_state, done))


  # Performs training step on a batch of timesteps from memory
  def train_batch_timesteps(self):

    # Get a sample if memory contains at least a batch size
    if len(self.memory) > StateAgent.BATCH_SIZE:
      mini_sample = random.sample(self.memory, StateAgent.BATCH_SIZE)
    else:
      mini_sample = self.memory # Otherwise just take the entire memory as a batch

    # Convert mini_sample into a list of state_features, actions, rewards, & dones
    prev_states_features, actions, rewards, curr_states_features, dones = zip(*mini_sample)

    self.trainer.train_step(prev_states_features, actions, rewards, curr_states_features, dones)


  # Performs training step on a single timestep
  def train_immediate_timestep(self, prev_state_features, action, reward, curr_state_features, done):
    self.trainer.train_step(prev_state_features, action, reward, curr_state_features, done)


  # Gets an action tensor for a timestep
  def get_action_tensor(self, state_features):
    
    epsilon = ACTION_TENSOR_EPSILON_THRESHOLD - self.n_games

    # Either take a random action (exploration)
    # or prediction action from the model (exploitation).
    # Likeihood of random action decays as number of games increases
    if random.randint(0, 10) < epsilon:
        action_tensor = self.get_random_action()
    else:
        action_tensor = self.model(state_features)

    return action_tensor
  

  # Use thresholding for boolean actions
  def get_random_action(self):
    acceleration = random.random() # between 0 to 1
    steer = random.uniform(-1, 1)
    brake = random.random() > 0.5
    drift = random.random() > 0.5
    fire = random.random() > 0.5
    nitro = random.random() > 0.5

    return torch.tensor((acceleration,
                        steer,
                        brake,
                        drift,
                        fire,
                        nitro),
                      device=DEVICE)


class Team:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt'))


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

          player_features = player_features.to(DEVICE)

          # Get network output by forward feeding features through the model
          network_output = self.model(player_features)

          # add action dictionary to actions list
          actions.append(get_action_dictionary_from_network_output(network_output))

        return actions


def get_features(player_state, team_state, opponent_states, puck_state, team_id):

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
    features_dict["opponent_goal_line_center"] = get_team_goal_line_center(puck_state, get_goal_by_team_id(team_id+1))
    features_dict["puck_to_opponent_goal_line_angle"] = get_obj1_to_obj2_angle(features_dict["puck_center"], features_dict["opponent_goal_line_center"])
    features_dict["kart_to_opponent_goal_line_difference"] = get_obj1_to_obj2_angle_difference(features_dict["player_kart_angle"], features_dict["puck_to_opponent_goal_line_angle"])

    # - Team goal line
    features_dict["team_goal_line_center"] = get_team_goal_line_center(puck_state, get_goal_by_team_id(team_id))
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

    return torch.tensor(features_list, dtype=torch.float32)

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

def get_team_goal_line_center(puck_state, team_id):
  return torch.tensor(puck_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

def get_team_goal_line(puck_state, team_id):
  return torch.tensor(puck_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]]

# limit angle between -1 to 1
def limit_period(angle):
  return angle - torch.floor(angle / 2 + 0.5) * 2 

def get_obj1_to_obj2_angle_difference(object1_angle, object2_angle):
  angle_difference = (object1_angle - object2_angle) / np.pi
  return limit_period(angle_difference)

# Assumes there are always two teams (0 and 1)
def get_goal_by_team_id(team_id):
  return (team_id+1) % 2



# TODO: Make this reward function more robust
def get_reward(player_state, team_state, opponent_states, puck_state, team_id):
  DEBUG_EN = True
  reward = 0

  player_goal_line = get_team_goal_line(puck_state, get_goal_by_team_id(team_id))
  opponent_goal_line = get_team_goal_line(puck_state, get_goal_by_team_id(team_id+1))
  puck_center = get_puck_center(puck_state)
  kart_center = get_kart_center(player_state)

  #if DEBUG_EN:
  #  print("get_reward - puck_center ", puck_center, " opponent goal line ", opponent_goal_line)
  #  print("get_reward - puck_center ", puck_center, " player goal line ", player_goal_line)

  if is_touching(kart_center, puck_center):
    reward += 10
    if DEBUG_EN:
      print("player is touching the puck")

  if is_puck_in_goal(puck_center, player_goal_line):
    reward -= 100
    if DEBUG_EN:
      print("puck is in player goal")
  elif is_puck_in_goal(puck_center, opponent_goal_line):
    reward += 100
    if DEBUG_EN:
      print("puck is in opponent goal")
  
  return torch.tensor(reward, dtype=torch.float32)

def is_touching(object1_center, object2_center, threshold=OBJS_TOUCHING_DISTANCE_THRESHOLD):
  return get_obj1_to_obj2_distance(object1_center, object2_center) < threshold

def is_puck_in_goal(puck_center, goal_line):
  # The puck is in goal if the puck_center is in between the two sides of the goal
  # and past the Y axis of the goal line minus a small buffer
  
  if goal_line[0][0] < puck_center[0] < goal_line[1][0] and \
  abs(puck_center[1]) >= abs(goal_line[0][1])-GOAL_LINE_Y_BUFFER:
    return True
  else:
    return False

def get_obj1_to_obj2_distance(object1_center, object2_center):
  return F.pairwise_distance(object1_center, object2_center)


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

# Get the score of a team using team_id
def get_score(puck_state, team_id):
  return puck_state["score"][get_goal_by_team_id(team_id)]
