import os

from state_agent.core_utils import load_model
from .agents import ComposedAgent, ComposedAgentNetwork
from .actors import SteeringActor, SpeedActor, DriftActor
from .planners import PlayerPuckGoalPlannerActor, PlayerPuckGoalFineTunedPlannerActor

# save a composed agent
agent = ComposedAgent()

steering_actor = SteeringActor()
speed_actor = SpeedActor()
drift_actor = DriftActor()
planner_actor = PlayerPuckGoalPlannerActor()
ft_planner_actor = PlayerPuckGoalFineTunedPlannerActor()
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
print(load_model(model=steering_actor.actor_net, model_name="steer_net", load_path=path))
print(load_model(model=speed_actor.actor_net, model_name="speed_net", load_path=path))
print(load_model(model=drift_actor.actor_net, model_name="drift_net", load_path=path))
print(load_model(model=planner_actor.actor_net, model_name="planner_net", load_path=path))
print(load_model(model=ft_planner_actor.actor_net, model_name="ft_planner_net", load_path=path))

#planner_actor.save_model(use_jit=True)

#steering_actor.save_model(use_jit=True)
#speed_actor.save_model(use_jit=True)
#drift_actor.save_model(use_jit=True)

# load a composed agent
#composed_agent = ComposedAgent()
#composed_agent.load_models("agent_basic_net2", use_jit=True)

#agent = ComposedAgent()
#agent.agent_net = composed_agent.agent_net
#agent.agent_net.speed_actor = composed_agent.agent_net.speed_actor
#agent.agent_net.drift_actor = composed_agent.agent_net.drift_actor
#agent.save_models("agent_basic_net2", use_jit=True)

#steering_actor = SteeringActor()
#speed_actor = SpeedActor()
#drift_actor = DriftActor()
#steering_actor.load_model()
#speed_actor.load_model()
#drift_actor.load_model()

composed_network = ComposedAgentNetwork(
    steering_actor.actor_net,
    speed_actor.actor_net,
    drift_actor.actor_net,
    planner_actor.actor_net,
    ft_planner_actor.actor_net
)

agent = ComposedAgent(composed_network)
agent.save_models("agent_net", use_jit=True)

