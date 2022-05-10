from .agents import ComposedAgent, ComposedAgentNetwork
from .actors import SteeringActor, SpeedActor, DriftActor

# save a composed agent
agent = ComposedAgent()

steering_actor = SteeringActor()
speed_actor = SpeedActor()
drift_actor = DriftActor()
print(steering_actor.load_model())
print(speed_actor.load_model())
print(drift_actor.load_model())

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
    steering_actor.actor_net,   # type: ignore
    speed_actor.actor_net,     # type: ignore
    drift_actor.actor_net      # type: ignore
    #None,  # type: ignore
    #None   # type: ignore
)

agent = ComposedAgent(composed_network)
agent.save_models("agent_basic_net", use_jit=True)

