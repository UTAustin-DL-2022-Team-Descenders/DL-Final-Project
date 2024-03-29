# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

from typing import List
import torch

if "1.9" not in torch.__version__:
    print("WARNING! Submission grader is using a different torch version than locally installed! Use 1.9.x")

LEGAL_KART_NAMES = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue', 'tux']

# Fix Team's player karts or Agent target speeds. Randomized if not given
# The length of these lists must match num_of_players (== 2)
TEAM_KART_LIST = ['hexley', 'sara_the_wizard']
AGENT_TARGET_SPEED = [16.0, 19.5]
USE_FINE_TUNED_PLANNER = [True, True]

class Team():
    agent_type = 'state'
    def __init__(
            self,
            num_of_players=2,
            train=False            
    ):
        from .agents import BaseAgent
        # From BaseTeam
        self.team = None
        self.num_players = num_of_players
        self.training_mode = None
        self.agents: List[BaseAgent] = []

        # Fine tuned planner settings
        self.use_fine_tuned_planner = USE_FINE_TUNED_PLANNER

        # List of target agent speeds
        self.agent_target_speed_list = AGENT_TARGET_SPEED

        # List of karts on this team
        self.team_kart_list = TEAM_KART_LIST

        # Check that team_kart_list are legal kart names
        if not set(self.team_kart_list).issubset(set(LEGAL_KART_NAMES)):
            raise Exception("At least one of the carts is not defined: ", self.team_kart_list)
        
    def set_training_mode(self, mode):
        self.training_mode = mode

    def new_match(self, team: int, num_players: int) -> list:

        self.team, self.num_players = team, num_players
        self.time_step = 0

        use_jit = True

        for agent in self.agents:
            agent.reset()

        if use_jit:
            self.agents = [
                self.create_composed_network("agent_net", # use a different name for agent 1 vs agent 2 based on the configured actors
                    target_speed=self.agent_target_speed_list[0],
                    use_finetuned_planner=self.use_fine_tuned_planner[0]
                ),
                self.create_composed_network("agent_net",
                    target_speed=self.agent_target_speed_list[1],
                    use_finetuned_planner=self.use_fine_tuned_planner[1]
                )
            ]
        else:
            self.create_discrete_networks()

        return self.team_kart_list

    def act(self, player_states, opponent_states, soccer_state):

        from .core_utils import DictObj
        
        # actions list for all players on this team
        actions = []

        # Iterate over all players on this team
        for player_num, player_state in enumerate(player_states):
            # Get network output by forward feeding features through the model
            output = dict(vars(self.agents[player_num](DictObj(player_state['kart']), DictObj(soccer_state), self.team)))

            # add action dictionary to actions list
            actions.append(output)

        self.time_step += 1

        return actions


    def create_discrete_networks(self):
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor
        from .agents import BaseTeam, Agent

        self.steering_actor = SteeringActor()
        self.steering_actor.load_model(use_jit=True)

        self.speed_actor = SpeedActor()
        self.speed_actor.load_model(use_jit=True)

        self.drift_actor = DriftActor()
        self.drift_actor.load_model(use_jit=True)

        self.planner_actor = PlayerPuckGoalPlannerActor()
        self.planner_actor.load_model(use_jit=True)

        self.ft_planner_actor = PlayerPuckGoalFineTunedPlannerActor()
        self.ft_planner_actor.load_model(use_jit=True)


        self.agents = [
            Agent(
                self.planner_actor,
                self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=self.agent_target_speed_list[0]
            ),
            Agent(
                self.planner_actor,
                self.ft_planner_actor,
                self.steering_actor,
                self.speed_actor,
                self.drift_actor,
                target_speed=self.agent_target_speed_list[1]
            )
        ]


    def create_composed_network(self, model_name,
        target_speed=23.0,
        use_drift=True,
        use_steer=True,
        use_speed=True,
        use_finetuned_planner=True,
        use_planner=True
    ):
        from .agents import ComposedAgent

        agent = ComposedAgent(
            None,
            target_speed=target_speed,
            use_drift=use_drift,
            use_speed=use_speed,
            use_steering=use_steer,
            use_planner=use_planner,
            use_finetuned_planner=use_finetuned_planner
        )
        agent.load_models(model_name, use_jit=True)

        return agent

    def save_composed_network(self, model_name):
        from .agents import BaseTeam, Agent, ComposedAgent, ComposedAgentNetwork
        from .actors import SteeringActor, SpeedActor, DriftActor
        from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor

        planner_actor = PlayerPuckGoalPlannerActor()
        planner_actor.load_model(use_jit=True)

        steering_actor = SteeringActor()
        steering_actor.load_model(use_jit=True)

        speed_actor = SpeedActor()
        speed_actor.load_model(use_jit=True)

        drift_actor = DriftActor()
        drift_actor.load_model(use_jit=True)

        planner_actor = PlayerPuckGoalPlannerActor()
        planner_actor.load_model(use_jit=True)

        ft_planner_actor = PlayerPuckGoalFineTunedPlannerActor()
        ft_planner_actor.load_model(use_jit=True)

        composed_network = ComposedAgentNetwork(
            steering_actor.actor_net,
            speed_actor.actor_net,
            drift_actor.actor_net,
            planner_actor.actor_net,
            ft_planner_actor.actor_net
        )

        agent = ComposedAgent(composed_network)
        agent.save_models(model_name, use_jit=True)
        print(agent.load_models(model_name, use_jit=True).state_dict())

        return agent
