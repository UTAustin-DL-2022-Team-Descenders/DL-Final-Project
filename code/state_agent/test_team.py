from typing import List
import torch
from .team import Team, LEGAL_KART_NAMES

TEAM_KART_LIST = []
AGENT_TARGET_SPEED = []
USE_FINE_TUNED_PLANNER = []

class TestTeam(Team):
    agent_type = 'state'
    def __init__(
            self,
            num_of_players=2,
            train=False,
            time_act_func=False,
            team_kart_list=None,
            agent_target_speed_list=None
    ):
        from .agents import BaseAgent
        import random

        super().__init__(num_of_players=num_of_players, train=train)

        # From BaseTeam
        self.time_act_func = time_act_func  # Set flag to denote if we're timing the act function or not
        self.slowest_act_time = [0, 0, 0, 0, 0]  # set slowest time to act for tracking worst case
        self.time_step = 0 # time step during each match
        self.agent_target_speed_list = agent_target_speed_list  # List of target agent speeds
        self.team_kart_list = team_kart_list  # List of karts on this team

        # Fine tuned planner settings
        self.use_fine_tuned_planner = USE_FINE_TUNED_PLANNER

        # Add random fine tune if one wasn't given for all num_players
        for i in range(self.num_players - len(self.use_fine_tuned_planner)):
            use_fine_tuned = bool(round(random.uniform(0, 1)))
            self.use_fine_tuned_planner.append(use_fine_tuned)

        # List of target agent speeds
        self.agent_target_speed_list = agent_target_speed_list if agent_target_speed_list else AGENT_TARGET_SPEED

        # Add random agent target speeds if one wasn't given for all num_players
        for i in range(self.num_players - len(self.agent_target_speed_list)):
            agent_target_speed = round(random.uniform(0, 20)) / 2 + 12.0 # (12, 22, 0.5)
            self.agent_target_speed_list.append(agent_target_speed)

        # List of karts on this team
        self.team_kart_list = team_kart_list if team_kart_list else TEAM_KART_LIST

        # Add random karts to team_kart_list if one wasn't given for all num_players
        for i in range(self.num_players - len(self.team_kart_list)):
            kart_name = random.choice(LEGAL_KART_NAMES)
            self.team_kart_list.append(kart_name)

        # Check that team_kart_list are legal kart names
        if not set(self.team_kart_list).issubset(set(LEGAL_KART_NAMES)):
            raise Exception("At least one of the carts is not defined: ", self.team_kart_list)

        # Print information about this Team
        print(f"Team - carts in use:", end=" ")
        for i in range(num_of_players):
            print(f"{self.team_kart_list[i]} using speed {self.agent_target_speed_list[i]:.1f}, fine-tuned: {self.use_fine_tuned_planner[i]}", end="; ")
        print("\n", end="")

    def act(self, player_states, opponent_states, soccer_state):

        import time

        # Collect start time if timeit is set
        if self.time_act_func:
            start_time = time.time()

        actions = super().act(player_states=player_states, opponent_states=opponent_states, soccer_state=soccer_state)

        # Print act execute time if timeit is set
        if self.time_act_func:
            end_time = time.time()
            act_time = end_time-start_time

            # Print only the slowest act executions
            if act_time > min(self.slowest_act_time):
                # pop the fastest time and add a slower time
                self.slowest_act_time.append(round(act_time * 1000, 1))
                self.slowest_act_time.pop(0)
                self.slowest_act_time.sort()
                if self.slowest_act_time[0] != 0:
                    print(f'Team.act slowest five acts in {self.slowest_act_time}ms (step: {self.time_step}')

            # Print act execution every timestep.
            # WARNING: adds huge number of print statements
            #print(f'Team.act in {(act_time*1000):.1f}ms, step: {self.time_step}')

        return actions

