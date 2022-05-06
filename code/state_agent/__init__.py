import os
if not os.getenv("TRAINING"):
    from state_agent.agents.subnets.modules.final import Team
