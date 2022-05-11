from .runner import TeamRunner, Match, MatchException
from .grader import Grader, Case
import random
import numpy as np

STEPS_PER_MATCH = 1200
MAX_TIME_IMAGE = 0.05 * STEPS_PER_MATCH
MAX_TIME_STATE = 0.01 * STEPS_PER_MATCH

# Flag to Randomize Ball starting location
RANDOMIZE_BALL_LOCATION = True

# Fix Team's player karts or Agent target speeds. Randomized if empty
# The length of these lists must match num_of_players (== 2)
TEAM_KART_LIST = ["tux", "tux"] # fixed
#TEAM_KART_LIST = [] # randomized

AGENT_TARGET_SPEED = [21.0, 12.0] # fixed
#AGENT_TARGET_SPEED = [] # randomized

# Print the Team act execution
PRINT_TEAM_ACT_EXECUTION = True

class HockyRunner(TeamRunner):
    """
        Similar to TeamRunner but this module takes Team object as inputs instead of the path to module
    """
    def __init__(self, team):
        self._team = team
        self.agent_type = self._team.agent_type


class FinalGrader(Grader):
    """Match against Instructor/TA's agents"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_model = HockyRunner(self.module.Team(team_kart_list=TEAM_KART_LIST, agent_target_speed_list=AGENT_TARGET_SPEED, time_act_func=PRINT_TEAM_ACT_EXECUTION))
        self.match = Match(use_graphics=self.student_model.agent_type == 'image')

    def _test(self, agent_name):
        time_limit = MAX_TIME_STATE if self.student_model.agent_type == 'state' else MAX_TIME_IMAGE

        test_model = TeamRunner(agent_name)
        ball_locations = [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
        ]

        # Randomize ball locations if flag is set
        if RANDOMIZE_BALL_LOCATION:
            ball_locations = np.random.uniform(-1, 1, (4, 2))

        scores = []
        results = []
        try:
            for bl in ball_locations:
                result = self.match.run(self.student_model, test_model, 2, STEPS_PER_MATCH, max_score=3, initial_ball_location=bl, initial_ball_velocity=[0, 0], record_fn=None, timeout=time_limit, verbose=self.verbose)
                scores.append(result[0])
                results.append(f'{result[0]}:{result[1]}')

            for bl in ball_locations:
                result = self.match.run(test_model, self.student_model, 2, STEPS_PER_MATCH, max_score=3, initial_ball_location=bl, initial_ball_velocity=[0, 0], record_fn=None, timeout=time_limit, verbose=self.verbose)
                scores.append(result[1])
                results.append(f'{result[1]}:{result[0]}')
        except MatchException as e:
            print('Match failed', e.score)
            print(' T1:', e.msg1)
            print(' T2:', e.msg2)
            assert 0
        return sum(scores), results, ball_locations

    @Case(score=25)
    def test_geoffrey(self):
        """geoffrey agent"""
        scores, results, ball_locations = self._test('geoffrey_agent')
        return min(scores / len(results), 1), f"{scores} goals scored in {len(results)} games ({'  '.join(results)}) using starting locations {ball_locations}"

@Case(score=25)
    def test_yann(self):
        """yann agent"""
        scores, results, ball_locations = self._test('yann_agent')
        return min(scores / len(results), 1), f"{scores} goals scored in {len(results)} games ({'  '.join(results)}) using starting locations {ball_locations}"

@Case(score=25)
    def test_yoshua(self):
        """yoshua agent"""
        scores, results, ball_locations = self._test('yoshua_agent')
        return min(scores / len(results), 1), f"{scores} goals scored in {len(results)} games ({'  '.join(results)}) using starting locations {ball_locations}"


@Case(score=25)
    def test_jurgen(self):
        """jurgen agent"""
        if self.student_model.agent_type == 'state':
            scores, results, ball_locations = self._test('jurgen_agent')
        else:
            scores, results, ball_locations = self._test('image_jurgen_agent')
            return min(scores / len(results), 1), f"{scores} goals scored in {len(results)} games ({'  '.join(results)}) using starting locations {ball_locations}"
