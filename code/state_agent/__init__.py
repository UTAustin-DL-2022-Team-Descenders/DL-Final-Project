# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/23/2022

import os

if 'GRADER_TESTING' in os.environ:
    from state_agent.test_team import TestTeam as Team
else:
    from state_agent.team import Team