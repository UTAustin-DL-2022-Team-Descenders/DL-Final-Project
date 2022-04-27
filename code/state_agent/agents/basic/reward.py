from .state_agent import *
import torch.nn.functional as F

OBJS_TOUCHING_DISTANCE_THRESHOLD = 4 # Pixel distance threshold to denote objects are touching
GOAL_LINE_Y_BUFFER = 2 # Pixel distance buffer to denote the puck is in a goal (to ensure reward is observed)

# TODO: Make this reward function more robust
def get_reward(curr_state_dictionaries, next_state_dictionaries, actions, team_num, player_num):

    DEBUG_EN = True
    reward = torch.tensor(0, dtype=torch.float32, device=DEVICE)

    # Get agent & opponent dictionary key info
    player_team_key = "team%0d_state" % team_num
    opponent_team_num = 2 if team_num == 1 else 1
    opponent_team_key = "team%0d_state" % opponent_team_num

    player_goal_line_index = get_goal_line_index_from_team_num(team_num)
    opponent_goal_line_index = get_goal_line_index_from_team_num(opponent_team_num)

    # Get information for current state
    curr_player_and_team_states = curr_state_dictionaries[player_team_key]
    curr_player_state = curr_player_and_team_states[player_num]
    curr_team_state = curr_player_and_team_states[:player_num] + curr_player_and_team_states[player_num+1:]
    curr_opponent_states = curr_state_dictionaries[opponent_team_key]

    curr_kart_center = get_kart_center(curr_player_state)
    curr_kart_velocity = get_kart_velocity(curr_player_state)

    curr_puck_state = curr_state_dictionaries["soccer_state"]
    curr_puck_center = get_puck_center(curr_puck_state)

    curr_opponent_goal_center = get_team_goal_line_center(curr_puck_state, opponent_goal_line_index)
    curr_player_goal_center = get_team_goal_line_center(curr_puck_state, player_goal_line_index)

    # Get information for next state
    next_player_and_team_states = next_state_dictionaries[player_team_key]
    next_player_state = next_player_and_team_states[player_num]
    next_team_state = next_player_and_team_states[:player_num] + next_player_and_team_states[player_num+1:]
    next_opponent_states = next_state_dictionaries[opponent_team_key]

    next_kart_center = get_kart_center(next_player_state)
    next_kart_velocity = get_kart_velocity(next_player_state)

    next_puck_state = next_state_dictionaries["soccer_state"]
    next_puck_center = get_puck_center(next_puck_state)

    next_opponent_goal_center = get_team_goal_line_center(next_puck_state, opponent_goal_line_index)
    next_player_goal_center = get_team_goal_line_center(next_puck_state, player_goal_line_index)

    # Positive reward if player is getting closer to the puck
    # Negative reward if player is getting further away from the puck
    curr_kart_to_puck_distance = get_obj1_to_obj2_distance(curr_kart_center, curr_puck_center)
    next_kart_to_puck_distance = get_obj1_to_obj2_distance(next_kart_center, next_puck_center)
    if next_kart_to_puck_distance < curr_kart_to_puck_distance:
        reward += 1 
    elif next_kart_to_puck_distance > curr_kart_to_puck_distance:
        reward -= 1

    # Positive reward if puck is getting closer to the opponent goal
    # Negative reward if puck is getting further away from the opponent goal
    curr_puck_to_opponent_goal_distance = get_obj1_to_obj2_distance(curr_opponent_goal_center, curr_puck_center)
    next_puck_to_opponent_goal_distance = get_obj1_to_obj2_distance(next_opponent_goal_center, next_puck_center)
    if next_puck_to_opponent_goal_distance < curr_puck_to_opponent_goal_distance:
        reward += 5
    elif next_puck_to_opponent_goal_distance > curr_puck_to_opponent_goal_distance:
        reward -= 5

    # Increase reward as puck gets closer to the opponent goal
    #reward += (1/puck_to_opponent_goal_distance)*10

    # Decrease reward as puck get closer to player goal
    #reward -= (1/puck_to_player_goal_distance)*10

    # Increase reward as kart gets closer to the puck
    #reward += (1/kart_to_puck_distance)*10

    if is_touching(curr_kart_center, curr_puck_center):
        reward += 10
        if DEBUG_EN:
            print("player is touching the puck")

    if is_puck_in_team_goal(curr_puck_state, team_num):
        reward -= 100
        if DEBUG_EN:
            print("puck is in agent goal")

    if is_puck_in_team_goal(curr_puck_state, opponent_team_num):
        reward += 100
        if DEBUG_EN:
            print("puck is in opponent goal")

    return reward

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