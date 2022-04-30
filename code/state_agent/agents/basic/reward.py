from .state_agent import *
import torch.nn.functional as F

OBJS_TOUCHING_DISTANCE_THRESHOLD = 4 # Pixel distance threshold to denote objects are touching
GOAL_LINE_Y_BUFFER = 2 # Pixel distance buffer to denote the puck is in a goal (to ensure reward is observed)
POINTED_TOWARDS_OBJ_THRESHOLD = 0.05

ACCELERATION_INDEX = 0
STEERING_INDEX = 1
BRAKE_INDEX = 2

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
    curr_kart_front = get_kart_front(curr_player_state)
    curr_kart_velocity = get_kart_velocity(curr_player_state)
    curr_kart_angle = get_obj1_to_obj2_angle(curr_kart_center, curr_kart_front)

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
    next_kart_front = get_kart_front(next_player_state)
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
    #elif next_kart_to_puck_distance > curr_kart_to_puck_distance:
    else:
        reward -= 1

    # Positive reward if puck is getting closer to the opponent goal
    # Negative reward if puck is getting further away from the opponent goal
    curr_puck_to_opponent_goal_distance = get_obj1_to_obj2_distance(curr_opponent_goal_center, curr_puck_center)
    next_puck_to_opponent_goal_distance = get_obj1_to_obj2_distance(next_opponent_goal_center, next_puck_center)


    # Get a reward if we're moving toward the puck
    if is_kart_pointed_towards_puck(curr_player_state, curr_puck_state):
      reward += 1
      # And extra reward if we're really moving at the puck
      if actions[ACCELERATION_INDEX] > 0 and \
       abs(actions[STEERING_INDEX]) < POINTED_TOWARDS_OBJ_THRESHOLD and \
         actions[BRAKE_INDEX] < 0.5:
        reward += 5
    elif actions[ACCELERATION_INDEX] == 0 and actions[BRAKE_INDEX] < 0.5:
      reward -= 5

      #curr_kart_direction = get_obj1_to_obj2_angle(curr_kart_center, curr_kart_front)
      #next_kart_direction = get_obj1_to_obj2_angle(next_kart_center, next_kart_front)
      #if get_obj1_to_obj2_angle_difference(curr_kart_direction, next_kart_direction) == 0:
      #  reward -= 5
      #else:
      #  reward -= 1

    # Get a reward if puck is moving toward a goal
    curr_puck_to_opponent_goal_angle = get_obj1_to_obj2_angle(curr_puck_center, curr_opponent_goal_center)
    if curr_puck_to_opponent_goal_angle == 0:
      #print("puck is heading towards the goal")
      reward += 1
      if is_touching(curr_kart_center, curr_puck_center):
        #print("player is causing the puck to go towards the goal!!")
        reward += 1
        if actions[ACCELERATION_INDEX] == 1:
          reward += 10
      

    if is_touching(curr_kart_center, curr_puck_center):
        reward += 10
        if DEBUG_EN:
            print("player is touching the puck")

    if is_puck_in_team_goal(curr_puck_state, team_num):
        reward -= 1000
        if DEBUG_EN:
            print("puck is in agent goal")

    if is_puck_in_team_goal(curr_puck_state, opponent_team_num):
        reward += 1000
        if DEBUG_EN:
            print("puck is in opponent goal")

    return reward

def is_touching(object1_center, object2_center, threshold=OBJS_TOUCHING_DISTANCE_THRESHOLD):
  return get_obj1_to_obj2_distance(object1_center, object2_center) < threshold


def is_kart_pointed_towards_puck(player_state, puck_state):

  player_kart_front = get_kart_front(player_state)
  player_kart_center = get_kart_center(player_state)
  puck_center = get_puck_center(puck_state)
  
  player_kart_angle = get_obj1_to_obj2_angle(player_kart_center, player_kart_front)
  kart_to_puck_angle = get_obj1_to_obj2_angle(player_kart_center, puck_center)
  kart_to_puck_angle_difference  = get_obj1_to_obj2_angle_difference(player_kart_angle, kart_to_puck_angle)

  return abs(kart_to_puck_angle_difference) < POINTED_TOWARDS_OBJ_THRESHOLD

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