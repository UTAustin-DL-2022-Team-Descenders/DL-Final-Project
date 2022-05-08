from collections import OrderedDict
from itertools import groupby
from operator import itemgetter
from random import shuffle

from state_agent.utils import check_reached_puck


class Movement:

    def __init__(self):
        self.frames = []
        self.team1_goal = [0, -64.5]
        self.team2_goal = [0, 64.5]
        self.unproc_frames = None

    def load_frame(self, player_states, opponent_states, soccer_state, actions):
        player1_location = [
            player_states[0]['kart']['location'][0],
            player_states[0]['kart']['location'][2]
        ]
        player2_location = [
            player_states[1]['kart']['location'][0],
            player_states[1]['kart']['location'][2]
        ]
        player1_velocity = [
            player_states[0]['kart']['velocity'][0],
            player_states[0]['kart']['velocity'][2]
        ]
        player2_velocity = [
            player_states[1]['kart']['velocity'][0],
            player_states[1]['kart']['velocity'][2]
        ]
        player1_quaternion = [
            player_states[0]['kart']['rotation'][1],
            player_states[0]['kart']['rotation'][3]
        ]
        player2_quaternion = [
            player_states[1]['kart']['rotation'][1],
            player_states[1]['kart']['rotation'][3]
        ]
        # Remember that these are the actions of the last timestamp, and the
        # locations are the result of the actions saved in the these dictionaries
        player1_actions = {
            'accel': actions[0]['acceleration'].item(),
            'steer': actions[0]['steer'].item(),
            'brake': actions[0]['brake'].item()
        }
        player2_actions = {
            'accel': actions[2]['acceleration'].item(),
            'steer': actions[2]['steer'].item(),
            'brake': actions[2]['brake'].item()
        }
        score = soccer_state['score']
        puck_location = [
            soccer_state['ball']['location'][0],
            soccer_state['ball']['location'][2]
        ]
        opponent_player1_location = [
            opponent_states[0]['kart']['location'][0],
            opponent_states[0]['kart']['location'][2]
        ]
        self.frames.append(dict(
            player1_location=player1_location,
            player2_location=player2_location,
            player1_velocity=player1_velocity,
            player2_velocity=player2_velocity,
            player1_quaternion=player1_quaternion,
            player2_quaternion=player2_quaternion,
            player1_actions=player1_actions,
            player2_actions=player2_actions,
            score=score,
            puck_location=puck_location,
            opponent_player1_location=opponent_player1_location
        ))

    def update_processed(self, indeces):
        for i in indeces:
            _ = self.unproc_frames.pop(i)

    def categorize(self, balance_moving=False):
        self.unproc_frames = OrderedDict()
        for i in range(len(self.frames)):
            self.unproc_frames[i] = self.frames[i]

        frames_heading_to_puck, indeces_heading_to_puck \
            = self.category_heading_to_puck()
        self.update_processed(indeces_heading_to_puck)

        frames_puck_reset, indeces_puck_reset = self.category_puck_reset()
        self.update_processed(indeces_puck_reset)

        (frames_team_making_a_goal, indeces_team_making_a_goal,
         frames_opponent_making_a_goal, indeces_opponent_making_a_goal) \
            = self.cateogory_making_a_goal()
        # Weird cases where a goal is made and the goal score does remain
        # consistent over the frames after the goal, return None for the dict
        if frames_team_making_a_goal is None:
            return None
        self.update_processed(indeces_team_making_a_goal)  # TODO
        self.update_processed(indeces_opponent_making_a_goal)

        frames_stalled_agent, indeces_stalled_agent = self.category_stalled_agent()
        self.update_processed(indeces_stalled_agent)

        frames_backwards, indeces_backwards = self.category_backwards()
        self.update_processed(indeces_backwards)

        if balance_moving:
            self.balance_turns()

        return dict(
            frames_heading_to_puck=frames_heading_to_puck,
            indeces_heading_to_puck=indeces_heading_to_puck,
            frames_team_making_a_goal=frames_team_making_a_goal,
            indeces_team_making_a_goal=indeces_team_making_a_goal,
            frames_opponent_making_a_goal=frames_opponent_making_a_goal,
            indeces_opponent_making_a_goal=indeces_opponent_making_a_goal,
            frames_backwards=frames_backwards,
            indeces_backwards=indeces_backwards,
            frames_moving=self.unproc_frames,
            indeces_moving=list(self.unproc_frames)
        )

    """
    For all these categories the identification should be:
    1. series of frames that indicate in cateogry
    2. series of frames that indicate not in cateogry anymore
    """
    def category_heading_to_puck(self):
        reached_puck = False
        return_frames = []
        indeces = []
        for i in self.unproc_frames:
            frame = self.unproc_frames[i]
            # If there was a goal made reset reached_puck flag
            if frame['opponent_player1_location'][0] == 30 and reached_puck:
                reached_puck = False
            # If still heading to puck save the data
            if not reached_puck:
                reached_puck = check_reached_puck(
                    y=[frame['puck_location'][0],    0, frame['puck_location'][1]],
                    x0=[frame['player1_location'][0], 0, frame['player1_location'][1]],
                    x1=[frame['player2_location'][0], 0, frame['player2_location'][1]]
                )
                return_frames.append(frame)
                indeces.append(i)
        return return_frames, indeces

    def category_puck_reset(self, tol=0.6):
        return_frames = []
        pre_indeces = []
        indeces = []
        for i in self.unproc_frames:
            frame = self.unproc_frames[i]
            # If puck is reset to origin save indeces
            if (
                    -tol < frame['puck_location'][0] < tol
                    and -tol < frame['puck_location'][1] < tol
            ):
                pre_indeces.append(i)
        # Find all groups of consecutive numbers in the list. This allows us
        # to not include frames where the puck passes over the origin.
        for k, g in groupby(enumerate(pre_indeces), lambda ix: ix[0] - ix[1]):
            idx = list(map(itemgetter(1), g))
            if len(idx) > 10:
                indeces.extend(idx)
        for i in indeces:
            return_frames.append(self.unproc_frames[i])

        return return_frames, indeces

    def cateogory_making_a_goal(self, frames_before_goal=20):
        return_frames_team = []
        indeces_team = []
        return_frames_opponent = []
        indeces_opponent = []
        last_goal = [0, 0]
        idx_before_goal = None
        for i in self.unproc_frames:
            frame = self.unproc_frames[i]
            # If a goal was made, check that the previous frames_before_goal
            # are include the goal making
            if frame['score'] != last_goal:
                idx = list(range(
                    idx_before_goal - frames_before_goal + 1,  # 20 frames before
                    idx_before_goal+1))  # up to when the goal is made
                # If you can't collect frames_before_goal because they don't
                # exist, then try a smaller set before the goal was made (10).
                # This occurs if the agent starts a new match and immediately
                # scores.
                k = 0
                while (
                        not all(j in self.unproc_frames for j in idx)
                        and idx_before_goal - int(frames_before_goal / 2) + k != idx_before_goal + 1):
                    idx = list(range(
                        idx_before_goal - int(frames_before_goal / 2) + k,
                        idx_before_goal + 1))
                    k += 1
                assert all(j in self.unproc_frames for j in idx), (
                    'Desired frames to save before scoring goal is not part '
                    'of the unprocessed frames.'
                )
                # i == list(self.unproc_frames)[-1]
                # idx may not be the proper indeces of the goal making
                if (
                        i - 1 == idx_before_goal
                        and i != list(self.unproc_frames)[-1]
                ):
                    return None, None, None, None
                if frame['score'][0] - last_goal[0] == 1:  # Team 1 scored
                    indeces_team.extend(idx)
                elif frame['score'][1] - last_goal[1] == 1:  # Team 2 scored
                    indeces_opponent.extend(idx)
                last_goal = frame['score']
            idx_before_goal = i
        for i in indeces_team:
            return_frames_team.append(self.unproc_frames[i])
        for i in indeces_opponent:
            return_frames_opponent.append(self.unproc_frames[i])

        return (return_frames_team, indeces_team,
                return_frames_opponent, indeces_opponent)

    def category_stalled_agent(self, frames_stalled=10):
        return_frames = []
        indeces = []
        keys = list(self.unproc_frames)
        for i in range(len(keys)):
            if i + frames_stalled + 1 >= len(keys):
                break
            consecutive = all(keys[i+j+1] - keys[i+j] == 1 for j in range(frames_stalled))
            if not consecutive:
                continue
            stand_still = True
            # Check next ten frames to see if the puck has not moved
            for j in range(frames_stalled):
                f0 = self.unproc_frames[keys[i+j+1]]
                f1 = self.unproc_frames[keys[i+j]]
                if f0['puck_location'] != f1['puck_location']:
                    stand_still = False
                    break
            # If it has not moved then loop through all frames starting from i
            # and check when the puck starts to move again
            if stand_still:
                j = 0
                f0 = self.unproc_frames[keys[i+j+1]]
                f1 = self.unproc_frames[keys[i+j]]
                while f0['puck_location'] == f1['puck_location'] \
                        and keys[i+j+1] - keys[i+j] == 1\
                        and i+j != len(keys) - 2:
                    j += 1
                    f0 = self.unproc_frames[keys[i+j+1]]
                    f1 = self.unproc_frames[keys[i+j]]
                indeces.extend(list(range(keys[i], keys[i+j] + 2)))
                # If the puck as not moved since the end of the match then
                # break loop
                if indeces[-1] == keys[-1]:
                    break
        indeces = list(set(indeces))
        for i in indeces:
            return_frames.append(self.unproc_frames[i])
        return return_frames, indeces

    def category_backwards(self, frames_backwards=30):
        return_frames = []
        indeces = []
        keys = list(self.unproc_frames)
        backwards = []
        for i in range(len(keys) - 1):
            if keys[i+1] - keys[i] != 1:
                continue
            f0 = self.unproc_frames[keys[i]]
            f1 = self.unproc_frames[keys[i+1]]
            if (
                    f0['player1_actions']['accel'] == 0.0
                    and f0['player1_actions']['brake'] == 1
                    and f0['player2_actions']['accel'] == 0.0
                    and f0['player2_actions']['brake'] == 1
                    and f1['player1_actions']['accel'] == 0.0
                    and f1['player1_actions']['brake'] == 1
                    and f1['player2_actions']['accel'] == 0.0
                    and f1['player2_actions']['brake'] == 1
            ):
                backwards.append(keys[i])
        for k, g in groupby(enumerate(backwards), lambda ix: ix[0] - ix[1]):
            idx = list(map(itemgetter(1), g))
            if len(idx) > frames_backwards:
                indeces.extend(idx)
        for i in indeces:
            return_frames.append(self.unproc_frames[i])
        return return_frames, indeces

    def category_turning(self):
        raise NotImplementedError

    def cateogory_puck_against_wall(self):
        raise NotImplementedError

    def cateogory_playing_next_to_puck(self):
        raise NotImplementedError

    def category_puck_behind_agent(self):
        raise NotImplementedError

    def balance_turns(self):
        num_before = len(self.unproc_frames)
        both_left = []
        both_right = []
        both_straight = []
        one_straight_one_left = []
        one_straight_one_right = []
        for i in self.unproc_frames:
            frame = self.unproc_frames[i]
            steer1 = frame['player1_actions']['steer']
            steer2 = frame['player2_actions']['steer']

            if steer1 == -1 and steer2 == -1:
                both_left.append([i, frame])
            elif steer1 == 1 and steer2 == 1:
                both_right.append([i, frame])
            elif steer1 == 0 and steer2 == 0:
                both_straight.append([i, frame])
            elif (steer1 == -1 and steer2 == 0) or (steer1 == 0 and steer2 == -1):
                one_straight_one_left.append([i, frame])
            elif (steer1 == 1 and steer2 == 0) or (steer1 == 0 and steer2 == 1):
                one_straight_one_right.append([i, frame])

        num_left = len(both_left) * 2 + len(one_straight_one_left)
        num_right = len(both_right) * 2 + len(one_straight_one_right)
        num_left_straight = len(one_straight_one_left)
        num_right_straight = len(one_straight_one_right)
        num_straight = len(both_straight) * 2 + len(one_straight_one_left) + len(one_straight_one_right)

        shuffle(both_left)
        shuffle(both_right)

        # Assumes num_straight is always the least
        both_left_keep = both_left[:num_straight - num_left_straight]
        both_right_keep = both_right[:num_straight - num_right_straight]

        keep_shuff = both_straight + one_straight_one_left + one_straight_one_right + both_left_keep + both_right_keep
        keep_shuff.sort()
        self.unproc_frames = {}
        for group in keep_shuff:
            self.unproc_frames[group[0]] = group[1]

        num_after = len(self.unproc_frames)
        print(f'Balancing turns removed {num_before-num_after} frames')