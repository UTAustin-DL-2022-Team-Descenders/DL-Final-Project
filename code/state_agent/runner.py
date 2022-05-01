import sys
import logging
import numpy as np
import traceback
from collections import namedtuple
from functools import reduce

TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000

RunnerInfo = namedtuple('RunnerInfo', ['agent_type', 'error', 'total_act_time'])

def to_native(o):
    # Super obnoxious way to hide pystk
    import pystk
    _type_map = {pystk.Camera.Mode: int,
                 pystk.Attachment.Type: int,
                 pystk.Powerup.Type: int,
                 float: float,
                 int: int,
                 list: list,
                 bool: bool,
                 str: str,
                 memoryview: np.array,
                 property: lambda x: None}

    def _to(v):
        if type(v) in _type_map:
            return _type_map[type(v)](v)
        else:
            return {k: _to(getattr(v, k)) for k in dir(v) if k[0] != '_'}
    return _to(o)

def load_assignment(name, f_out=sys.stdout, pre_import_fn=None):
    import atexit
    import importlib
    from pathlib import Path
    from shutil import rmtree    
    import tempfile
    import zipfile

    if pre_import_fn is not None:
        pre_import_fn()
    return importlib.import_module(name)

class AIRunner:
    agent_type = 'state'
    is_ai = True

    def set_training_mode(self, mode):
        pass

    def new_match(self, team: int, num_players: int) -> list:
        pass

    def act(self, player_state, opponent_state, world_state):
        return []

    def info(self):
        return RunnerInfo('state', None, 0)


class TeamRunner:
    agent_type = 'state'
    _error = None
    _total_act_time = 0

    def __init__(self, team_or_dir):
        from pathlib import Path
        try:
            from grader import grader
        except ImportError:
            try:
                from . import grader
            except ImportError:
                import grader

        self._error = None
        self._team = None
        try:                 
            if isinstance(team_or_dir, (str, Path)):     
                assignment = load_assignment(team_or_dir)
                if assignment is None:
                    self._error = 'Failed to load submission.'
                else:                    
                    self._team = assignment.Team()
            else:                
                self._team = team_or_dir            
        except Exception as e:
            traceback.print_exc()
            self._error = Exception('Failed to load submission: {}'.format(str(e)))
        if hasattr(self, '_team') and self._team is not None:
            self.agent_type = self._team.agent_type

    def set_training_mode(self, mode):
        if hasattr(self._team, "set_training_mode"):
            self._team.set_training_mode(mode)

    def new_match(self, team: int, num_players: int) -> list:
        self._total_act_time = 0
        self._error = None
        try:      
            r = self._team.new_match(team, num_players)
            if isinstance(r, str) or isinstance(r, list) or r is None:
                return r
            self._error = Exception('new_match needs to return kart names as a str, list, or None. Got {!r}!'.format(r))
        except Exception as e:
            self._error = e
        return []

    def act(self, player_state, *args, **kwargs):
        from time import time
        t0 = time()
        try:            
            r = self._team.act(player_state, *args, **kwargs)
        except Exception as e:            
            self._error = e
        else:            
            self._total_act_time += time()-t0
            return r
        return []

    def info(self):
        return RunnerInfo(self.agent_type, self._error, self._total_act_time)
    
    def team(self):
        return self._team


class MatchException(Exception):
    def __init__(self, score, msg1, msg2, exp):
        self.exp = exp
        self.score, self.msg1, self.msg2 = score, msg1, msg2


class Match:
    """
        Do not create more than one match per process (use ray to create more)
    """
    def __init__(self, use_graphics=False, logging_level=None):
        # DO this here so things work out with ray
        import pystk
        self._pystk = pystk
        if logging_level is not None:
            logging.basicConfig(level=logging_level)

        # Fire up pystk
        self._use_graphics = use_graphics
        if use_graphics:
            graphics_config = self._pystk.GraphicsConfig.hd()
            graphics_config.screen_width = 400
            graphics_config.screen_height = 300
        else:
            graphics_config = self._pystk.GraphicsConfig.none()

        self._pystk.init(graphics_config)

    def __del__(self):
        if hasattr(self, '_pystk') and self._pystk is not None and self._pystk.clean is not None:  # Don't ask why...
            self._pystk.clean()

    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    @classmethod
    def _r(cls, f):
        if hasattr(f, 'remote'):
            return f.remote
        if hasattr(f, '__call__'):
            if hasattr(f.__call__, 'remote'):
                return f.__call__.remote
        return f

    @staticmethod
    def _g(f):
        from .remote import ray
        if ray is not None and isinstance(f, (ray.types.ObjectRef, ray._raylet.ObjectRef)):
            return ray.get(f)
        return f

    def _check(self, teams, where, n_iter, timeout):

        crash_idx = None

        timeouts = []
        errors = []
        for idx, team in enumerate(teams):
            _, error, t = self._g(self._r(team.info)())
            timeouts.append(t)
            errors.append(error)
            if error:
                crash_idx = idx
                break
                        
        if crash_idx is not None:
            raise errors[crash_idx]
            #raise MatchException(0, "", "", errors[crash_idx])

        logging.debug('timeout {} <? {}'.format(timeout, timeouts))
        return list(map(lambda t: t < timeout, timeouts))

    def run(self, teams, num_player=1, num_frames=MAX_FRAMES, max_score=3, record_fn=None, timeout=1e10,
            initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], training_mode=None, verbose=False):

        race = None
        RaceConfig = self._pystk.RaceConfig

        num_teams = len(teams)
        logging.info('Creating teams: number of teams {}, number of players on a team {}'.format(num_teams, num_player))

        # Start a new match
        team_cars = [self._g(self._r(team.new_match)(idx, num_player)) or ['tux'] * num_player for idx, team in enumerate(teams)]

        # set the training mode
        [self._g(self._r(team.set_training_mode)(training_mode)) for team in teams]

        for team in teams:
            if self._g(self._r(team.info)())[0] == "image":
                assert self._use_graphics, 'Need to use_graphics for image agents.'

        # Deal with crashes
        can_act = self._check(teams, 'new_match', 0, timeout)

        # Setup the race config
        logging.info('Setting up race')
        
        race_config = RaceConfig(track=TRACK_NAME, mode=RaceConfig.RaceMode.SOCCER, num_kart=num_teams * num_player)
        race_config.players.pop()
        for i in range(num_player):
            for team_i, team in enumerate(teams):
                is_ai = hasattr(team, 'is_ai') and team.is_ai
                race_config.players.append(self._make_config(team_i, is_ai, team_cars[team_i][i % len(team_cars)]))
                logging.info("    Player: {}, team: {}, is_ai {}".format(i, team_i, is_ai))    

        
        # Start the match
        logging.info('Starting race')
        
        race = self._pystk.Race(race_config)
        race.start()
        race.step()

        state = self._pystk.WorldState()
        state.update()
        state.set_ball_location((initial_ball_location[0], 1, initial_ball_location[1]),
                                (initial_ball_velocity[0], 0, initial_ball_velocity[1]))


        it = 0
        try:
        
            for step in range(num_frames):
                logging.debug('iteration {} / {}'.format(it, MAX_FRAMES))
                state.update()

                team_states = [[to_native(p) for p in state.players[i::num_teams]] for i in range(num_teams)]
                            
                soccer_state = to_native(state.soccer)
                team_images = [None for _ in teams]
                if self._use_graphics:                
                    team_images = [
                        [
                            np.array(race.render_data[i].image) 
                                for i in range(idx, len(race.render_data), 2)
                        ] 
                        for idx in range(len(teams))
                    ]   

                # Have each team produce actions (in parallel)
                team_actions_delayed = []
                for idx, (team, team_can_act, team_state, team_img) in enumerate(zip(teams, can_act, team_states, team_images)):            
                    if team_can_act:
                        if self._g(self._r(team.info)())[0] == "image":
                            team_actions_delayed.append(self._r(team.act)(team_state, team_img))
                        else:
                            team_actions_delayed.append(self._r(team.act)(team_state, team_states[0] if idx == 1 else (team_states[1] if len(team_states) > 1 else None), soccer_state))

                # Wait for the actions to finish
                team_actions = [self._g(action_delayed) if team_can_act else None for action_delayed, team_can_act in zip(team_actions_delayed, can_act)]
                
                new_can_act = self._check(teams, 'act', it, timeout)
                for idx, (act1, act2) in enumerate(zip(new_can_act, can_act)):
                    if not act1 and act2 and verbose:
                        print('Team {} timed out'.format(idx))
                    
                can_act = new_can_act

                # Assemble the actions
                actions = []
                for player_i in range(num_player):
                    for team_i, team_action in enumerate(team_actions):
                        actions.append(team_action[player_i] if team_action is not None and player_i < len(team_action) else {})                    
                    
                if record_fn:
                    self._r(record_fn)(team_states, soccer_state=soccer_state, actions=actions) # ignore team images for now, team_images=team_images)

                logging.debug('  race.step  [score = {}]'.format(state.soccer.score))
                if (not race.step([self._pystk.Action(**a) for a in actions]) and num_player) or sum(state.soccer.score) >= max_score:
                    print("Terminating match")
                    break

                it += 1

        finally:
            race.stop()
            del race

        return state.soccer.score

    def wait(self, x):
        return x

def runner(args):

    from pathlib import Path    
    from os import environ
    from state_agent import remote
    import state_agent.utils as utils
    
    teams = []
    team_agents = []
    states = []
    try:
    
        if args.parallel is None or remote.ray is None:
            # Create the teams
            if args.team1:
                teams.append(AIRunner() if args.team1 == 'AI' else TeamRunner(args.team1))
                team_agents.append(args.team1)
            if args.team2:
                teams.append(AIRunner() if args.team2 == 'AI' else TeamRunner(args.team2))
                team_agents.append(args.team2)
            
            use_graphics = reduce(lambda prev, team: team.agent_type == 'image' or prev, teams, False)
            match = Match(use_graphics=use_graphics)
            
            # Start the match
            for m in range(args.matches):
                
                # What should we record?
                recorder = None
                if args.record_video:
                    record_video_suffix = Path(args.record_video).suffix
                    record_video_name = args.record_video.replace(record_video_suffix, "")                    
                    recorder = utils.VideoRecorder("{}_{:05d}{}".format(record_video_name, m, record_video_suffix))

                # always record the state (this is in the state_agent module so it is a valid assumption)
                record_state_file = None
                if args.record_state:
                    record_state_suffix = Path(args.record_state).suffix                    
                    record_state_name = args.record_state.replace(record_state_suffix, "")  
                    record_state_file = "{}_{:05d}{}".format(record_state_name, m, record_state_suffix)
                # else save the states in memory

                recorder = recorder & utils.StateRecorder(record_state_file)
                    
                try:
                    result = match.run(teams, args.num_players, args.num_frames, max_score=args.max_score,
                                    initial_ball_location=args.ball_location, initial_ball_velocity=args.ball_velocity,
                                    training_mode=args.training_mode,
                                    record_fn=recorder)
                    print('Match results', result)                                                
                    states.append(recorder.states)                        
                except MatchException as e:
                    raise e.exp

            del match

            teams = [team._team if hasattr(team, "_team") else 'AI' for team in teams], states

        else:
            # Fire up ray
            remote.init(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()), configure_logging=True,
                        log_to_driver=True, include_dashboard=False)

            # Create the teams            
            team_infos = []            
            if args.team1:                                
                team1 = AIRunner() if args.team1 == 'AI' else remote.RayTeamRunner.remote(args.team1)
                teams.append(team1)
                team_agents.append(args.team1)
                team_infos.append(team1.info() if args.team1 == 'AI' else remote.get(team1.info.remote()))
            if args.team2:
                team2 = AIRunner() if args.team2 == 'AI' else remote.RayTeamRunner.remote(args.team2)
                teams.append(team2)           
                team_agents.append(args.team2) 
                team_infos.append(team2.info() if args.team2 == 'AI' else remote.get(team2.info.remote()))

            # What should we record?
            # assert args.record_state is None or args.record_video is None, "Cannot record both video and state in parallel mode"

            # Start the jobs
            matches = [
                remote.RayMatch.remote(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()),
                                        use_graphics=reduce(lambda prev, team: team[0] == "image" or prev, team_infos, False))
                for i in range(args.parallel)
            ]

            # Start the match
            results = []
            for i in range(args.matches):
                recorder = None
                # XXX difficult to make both work?
                #if args.record_video:
                #    ext = Path(args.record_video).suffix
                #    recorder = remote.RayVideoRecorder.remote(args.record_video.replace(ext, '_{:05d}{}'.format(i, ext)))
                #elif args.record_state:
                record_state_file = None
                if args.record_state:
                    ext = Path(args.record_state).suffix
                    record_state_file = args.record_state.replace(ext, '_{:05d}{}'.format(i, ext))
                    
                recorder = remote.RayStateRecorder.remote(record_state_file)

                result = matches[i % args.parallel].run.remote(teams, args.num_players, args.num_frames, max_score=args.max_score,
                                        initial_ball_location=args.ball_location,
                                        initial_ball_velocity=args.ball_velocity,
                                        training_mode=args.training_mode,
                                        record_fn=recorder)
                results.append((result, recorder))                

            for result, recorder in results:
                try:
                    result = remote.get(result)
                    recorder_states = remote.get(recorder.get_states.remote())
                    states.append(recorder_states)
                except (remote.RayMatchException, MatchException) as e:
                    print('Match failed', e.score)
                    print(' T1:', e.msg1)
                    print(' T2:', e.msg2)

                print('Match results', result)

            remote.shutdown()

    except Exception as e:
        traceback.print_exc()

    return team_agents, states

def main(args_local=None):
    import argparse 
    from os import environ
    
    parser = argparse.ArgumentParser(description="Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
    parser.add_argument('-r', '--record_video', help="Do you want to record a video?")
    parser.add_argument('-s', '--record_state', help="Do you want to pickle the state?")
    parser.add_argument('-f', '--num_frames', default=1200, type=int, help="How many steps should we play for?")
    parser.add_argument('-p', '--num_players', default=2, type=int, help="Number of players per team")
    parser.add_argument('-m', '--max_score', default=3, type=int, help="How many goal should we play to?")
    parser.add_argument('-j', '--parallel', type=int, help="How many parallel process to use?")
    parser.add_argument('--matches', default=1, type=int, help="Number of matches to run")
    parser.add_argument('--ball_location', default=[0, 0], type=float, nargs=2, help="Initial xy location of ball")
    parser.add_argument('--ball_velocity', default=[0, 0], type=float, nargs=2, help="Initial xy velocity of ball")
    parser.add_argument('team1', help="Python module name or `AI` for AI players.")
    parser.add_argument('--team2', default=None, help="Python module name or `AI` for AI players.")
    parser.add_argument('--training_mode', default=None, type=str, help="Training mode")
    args = parser.parse_known_args()[0]

    logging.basicConfig(level=environ.get('LOGLEVEL', 'WARNING').upper())

    if args_local:
        new_dict: dict = vars(args).copy()
        new_dict.update(vars(args_local))
        args = argparse.Namespace(**new_dict)

    return runner(args)        

if __name__ == '__main__':    
    main()