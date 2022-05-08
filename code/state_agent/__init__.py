# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

class Team():
    agent_type = 'state'
    def __init__(self, num_of_players=2, train=False):
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor
        from .agents import BaseTeam, Agent

        # From BaseTeam
        self.team = None
        self.num_players = 0
        self.training_mode = None

        self.steering_actor = SteeringActor()
        self.steering_actor.load_model(use_jit=True)

        self.speed_actor = SpeedActor()
        self.speed_actor.load_model(use_jit=True)

        self.drift_actor = DriftActor()
        self.drift_actor.load_model(use_jit=True)

        # TODO: how are planners initiated now?
        self.planner_actor = PlayerPuckGoalPlannerActor()
        self.planner_actor.load_model(use_jit=True)

        self.ft_planner_actor = PlayerPuckGoalFineTunedPlannerActor(mode="speed")
        self.ft_planner_actor.load_model(use_jit=True)

        # TODO: how is the Agent initiated now?
        self.agents = [
            Agent(
                self.planner_actor,
                self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=21.0
            ),
            Agent(
                self.planner_actor,
                # self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=12.0
            )
        ]

    def set_training_mode(self, mode):
        self.training_mode = mode

    def new_match(self, team: int, num_players: int) -> list:
        self.team, self.num_players = team, num_players
        for agent in self.agents:
            agent.reset()
        return ['tux'] * num_players

    def act(self, player_states, opponent_states, soccer_state):
        from .core_utils import DictObj
        # actions list for all players on this team
        actions = []

        # Iterate over all players on this team
        for player_num, player_state in enumerate(player_states):
            # Get network output by forward feeding features through the model
            output = self.agents[player_num](DictObj(player_state['kart']), DictObj(soccer_state), self.team)

            # add action dictionary to actions list
            actions.append(dict(vars(output)))

        return actions