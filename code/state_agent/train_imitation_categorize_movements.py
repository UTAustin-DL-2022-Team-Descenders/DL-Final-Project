import os.path
from operator import itemgetter
import pickle

from state_agent.utils import get_pickle_files
from tournament.utils import load_all_recording
from movement_categories.extract_features_for_categories import Movement


path = os.path.abspath(os.path.dirname(__file__))
training_pkl_file_list = get_pickle_files(os.path.join(path, '..', 'imitation_data'))
path_categorized_pkl = os.path.join(path, '..', 'categorized_data')

frames_heading_to_puck = []
frames_team_making_a_goal = []
frames_opponent_making_a_goal = []
frames_backwards = []
frames_moving = []

file = 0
for pkl_file in training_pkl_file_list:
    file += 1
    print(f'Loading {file} / {len(training_pkl_file_list)}: {pkl_file.split("/")[-1]}')
    frames = load_all_recording(pkl_file)
    mv = Movement()
    f = 0
    for states in frames:
        f += 1
        player_states = states["team1_state"]
        opponent_states = states["team2_state"]
        soccer_state = states["soccer_state"]
        actions = states["actions"]

        # if f % 10 == 0:
        #     print(
        #         f"{soccer_state['ball']['location'][0]:05.2f}, ",
        #         f"{soccer_state['ball']['location'][2]:05.2f} ||",
        #         f"{player_states[0]['kart']['location'][0]:05.2f}, ",
        #         f"{player_states[0]['kart']['location'][2]:05.2f}, ",
        #         f"{actions[0]['acceleration'].item():05.2f}, ",
        #         f"{actions[0]['steer'].item()}, ",
        #         f"{actions[0]['brake'].item()} ||",
        #         f"{player_states[1]['kart']['location'][0]:05.2f}, ",
        #         f"{player_states[1]['kart']['location'][2]:05.2f}, ",
        #         f"{actions[2]['acceleration'].item():05.2f}, ",
        #         f"{actions[2]['steer'].item()}, ",
        #         f"{actions[2]['brake'].item()}",
        #     )

        frames_new_start_to_puck = mv.load_frame(player_states, opponent_states, soccer_state, actions)

    sorted_frames = mv.categorize(balance_moving=True)

    if sorted_frames is None:
        print('Inconsistent score, skipping:', pkl_file.split('/')[-1])
        continue

    frames_heading_to_puck.extend(itemgetter(*sorted_frames['indeces_heading_to_puck'])(frames))
    if sorted_frames['indeces_moving']:
        frames_moving.extend(itemgetter(*sorted_frames['indeces_moving'])(frames))
    if sorted_frames['indeces_team_making_a_goal']:
        frames_team_making_a_goal.extend(itemgetter(*sorted_frames['indeces_team_making_a_goal'])(frames))
    if sorted_frames['indeces_opponent_making_a_goal']:
        frames_opponent_making_a_goal.extend(itemgetter(*sorted_frames['indeces_opponent_making_a_goal'])(frames))
    if sorted_frames['indeces_backwards']:
        frames_backwards.extend(itemgetter(*sorted_frames['indeces_backwards'])(frames))


with open(os.path.join(path_categorized_pkl, 'frames_heading_to_puck.pkl'), 'wb') as f:
    pickle.dump(frames_heading_to_puck, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(path_categorized_pkl, 'frames_moving.pkl'), 'wb') as f:
    pickle.dump(frames_moving, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(path_categorized_pkl, 'frames_team_making_a_goal.pkl'), 'wb') as f:
    pickle.dump(frames_team_making_a_goal, f, pickle.HIGHEST_PROTOCOL)
# with open(os.path.join(path_categorized_pkl, 'frames_opponent_making_a_goal.pkl'), 'wb') as f:
#     pickle.dump(frames_opponent_making_a_goal, f, pickle.HIGHEST_PROTOCOL)
# with open(os.path.join(path_categorized_pkl, 'frames_backwards.pkl'), 'wb') as f:
#     pickle.dump(frames_backwards, f, pickle.HIGHEST_PROTOCOL)
print('Done')
