import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import torch
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.utils.rendertools import RenderTool
#from .envs.flatland.utils.gym_env_wrappers import AvailableActionsWrapper, SkipNoChoiceCellsWrapper, SparseRewardWrapper, DeadlockWrapper, ShortestPathActionWrapper, DeadlockResolutionWrapper
from utils import observation_utils
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.core.grid.grid4_utils import get_new_position
from collections import defaultdict
from flatland.core.env_observation_builder import ObservationBuilder
from typing import Optional, List

from extras.envs.flatland.observations import Observation, register_obs, make_obs
@register_obs("combined")
class CombinedObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._observations = [
            make_obs(obs_name, config[obs_name]) for obs_name in config.keys()
        ]
        self._builder = CombinedObsForRailEnv([
            o._builder for o in self._observations
        ])

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        space = []
        for o in self._observations:
            space.append(o.observation_space())
        return gym.spaces.Tuple(space)


class CombinedObsForRailEnv(ObservationBuilder):

    def __init__(self, builders: [ObservationBuilder]):
        super().__init__()
        self._builders = builders

    def reset(self):
        for b in self._builders:
            b.reset()

    def get(self, handle: int = 0):
        return None

    def get_many(self, handles: Optional[List[int]] = None):
        obs = {h: [] for h in handles}
        for b in self._builders:
            sub_obs = b.get_many(handles)
            for h in handles:
                obs[h].append(sub_obs[h])
        return obs

    def set_env(self, env):
        for b in self._builders:
            b.set_env(env)

class FlatlandMultiAgentEnv(gym.Env):

    def __init__(self, cfg):
        # self.seed(1)
        self._deadlocked_agents = []

        self.cfg = cfg
        self._not_finished_reward = -2
        self._finished_reward = 2
        self._deadlock_reward = -2
        
        predictor = ShortestPathPredictorForRailEnv(cfg['observation_max_path_depth'])
        tree_observation = TreeObsForRailEnv(max_depth=cfg['observation_tree_depth'], predictor=predictor)

        self.env = RailEnv(
            width=cfg['x_dim'],
            height=cfg['y_dim'],
            rail_generator=sparse_rail_generator(
                max_num_cities=cfg['n_cities'],
                grid_mode=False,
                max_rails_between_cities=cfg['max_rails_between_cities'],
                max_rails_in_city=cfg['max_rails_in_city']
            ),
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=cfg['n_agents'],
            malfunction_generator_and_process_data=None,
            obs_builder_object=CombinedObsForRailEnv([tree_observation, GlobalObsForRailEnv()]),
            random_seed=None
        )
        
        if self.cfg['render']:
            self.renderer = RenderTool(self.env)
            self.renderer.set_new_rail()
            
        obs = self.reset()

        agent_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs['agents'][0].shape, dtype=np.float64)
        self.observation_space = spaces.Dict({
            'agents': spaces.Tuple((agent_obs_space,) * self.cfg['n_agents']),
            'state': spaces.Box(low=-100., high=100., shape=obs['state'].shape, dtype=np.float64),
        })

        agent_action_space = spaces.Discrete(len(RailEnvActions))
        self.action_space = spaces.Tuple((agent_action_space,) * self.cfg['n_agents'])
    
    def process_global_obs(self, obs):
        with torch.no_grad():
            o1,o2,o3 = torch.from_numpy(obs[0]).float().permute(2,0,1), torch.from_numpy(obs[1]).float().permute(2,0,1), torch.from_numpy(obs[2]).float().permute(2,0,1)
        return torch.cat((o1,o2,o3), 0).numpy()

    def make_global_obs(self, fl_obs):
        #Build global state
        observations_list = []
        
        valid_idx = 0
        for i, o in fl_obs.items():
            
            if o[1] is None:
                
                o_i = np.ones((23,25,25))*(-5)
                valid_idx+=1
            else:
                o_i = self.process_global_obs(o[1])
            observations_list.append(o_i)
 
        if valid_idx >= self.cfg["n_agents"]:
            valid_idx = 0 #this is only if all agents are done
            
            
        
        transitions = observations_list[valid_idx][0:16, :, :] # Get valid transitions at each location
        
        global_state  = np.concatenate((transitions, observations_list[valid_idx][18:19,:,:]), axis=0) # get all agents malfunctions
        global_state  = np.concatenate((global_state, observations_list[valid_idx][20:21,:,:]), axis=0) # get agents ready to depart from that position

        for obs in observations_list:
            global_state = np.concatenate((global_state, obs[16:17, :, :]), axis=0) #get agent's position and direction
            global_state = np.concatenate((global_state, obs[21:22, :, :]), axis=0) #get target positions
        # Based on docs: https://flatland.aicrowd.com/getting-started/env/observations.html
#        print(global_state.shape)
#        for i in  range(1, len(observations_list)):
#            global_observations = np.concatenate((global_observations, observations_list[i][16:, :, :]), axis=0)
        
#        print(global_observations.shape)
#        state = np.zeros(self.cfg['world_shape'] + [2], dtype=np.float64)

        return global_state


    def make_obs(self, fl_obs):
        obs_tree_depth = self.cfg['observation_tree_depth']
        obs_radius = self.cfg['observation_radius']

        obs = []
        for i, o in fl_obs.items():
            if o[0] is None:
                obs_i = np.zeros(self.observation_space['agents'][i].shape)
            else:
                obs_i = observation_utils.normalize_observation(o[0], obs_tree_depth, obs_radius)
            obs.append(obs_i)



        global_state = self.make_global_obs(fl_obs)

        state = np.zeros(self.cfg['world_shape'] + [2], dtype=np.float64)

        return {'agents': tuple(obs), 'state': global_state}

    def step(self, actions):
        action_dict = {i: a for i, a in enumerate(actions)}
        fl_obs, fl_reward, fl_done, fl_info = self.env.step(action_dict)


        reward = sum(fl_reward.values())
        obs = self.make_obs(fl_obs)
        
        r  =  {}
        comp  = {}
        deadlocked_agents = self.check_deadlock()
#        print(deadlocked_agents)
        for agent_id, agent_obs in fl_obs.items():
#            print(agent_id)
            if fl_done[agent_id]:
                if self.env.agents[agent_id].status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    
                    r[agent_id] = self._finished_reward
                    comp[agent_id] = True
                else:
                    
                    r[agent_id] = self._not_finished_reward
                    comp[agent_id] = False
            else:
                r[agent_id] = -2
                comp[agent_id] = False
            self.agent_scores[agent_id] += fl_reward[agent_id]
            self.agent_steps[agent_id] += 1

            if agent_id in deadlocked_agents:
                r[agent_id] += self._deadlock_reward
#                print("Agent {} deadlocked".format(agent_id))
        
        
        fl_reward = r

        done = fl_done['__all__']
        if done:
            print("Done!")
        info = {
            'rewards': fl_reward,
            'done': comp,
            'max_episode_steps': self.env._max_episode_steps,
            'num_agents': self.env.get_num_agents(),
            'agent_score': self.agent_scores,
            'agent_step': self.agent_steps,
            'agent_deadlock': self.deadlocks
            
        }
        return obs, reward, done, info

    # def seed(self, seed=None):
    #     self.random_state, seed = seeding.np_random(seed)
    #     return [seed]

    def check_deadlock(self):  # -> Set[int]:
        rail_env = self.env
        new_deadlocked_agents = []
        for agent in rail_env.agents:
#            print(agent)
            if agent.status == RailAgentStatus.ACTIVE and agent.handle not in self._deadlocked_agents:
                position = agent.position
                direction = agent.direction
                possible_transitions = rail_env.rail.get_transitions(*position, direction)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions == 1:
                    new_direction_me = np.argmax(possible_transitions)
                    new_cell_me = get_new_position(position, new_direction_me)
                    for opp_agent in rail_env.agents:

                        if new_cell_me is not None and opp_agent.position is not None:
                            if new_cell_me[0] == opp_agent.position[0] and new_cell_me[1] == opp_agent.position[1]:
                                opp_position = opp_agent.position
                                opp_direction = opp_agent.direction
                                opp_possible_transitions = rail_env.rail.get_transitions(*opp_position, opp_direction)
                                opp_num_transitions = np.count_nonzero(opp_possible_transitions)
                                if opp_num_transitions == 1: # check if the potential deadlocked agent has more than one possible transition
                                    if opp_direction != direction: # check if the deadlocked agents have the same direction. If so then no deadlock
                                        
                                        self.deadlock_counter[agent.handle]+=1
                                        if self.deadlock_counter[agent.handle] >= 5:
                                        
                                            self._deadlocked_agents.append(agent.handle) #add so they are not checked agaiin
                                            new_deadlocked_agents.append(agent.handle)
                                            self.deadlocks +=1


        return new_deadlocked_agents

    def render(self, mode='human', **kwargs):
        if self.cfg['render']:
            self.renderer.render_env(show=True, **kwargs)

    def reset(self):
        self.agent_scores = defaultdict(float)
        self.agent_steps = defaultdict(int)
        self._deadlocked_agents = []
        self.deadlocks = 0
        self.deadlock_counter = defaultdict()
        for i in range(self.cfg['n_agents']):
            self.deadlock_counter[i] = 0
        fl_obs, fl_info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)

        obs = self.make_obs(fl_obs)
        return obs

    def close(self):
        if self.cfg['render']:
            self.renderer.close_window()
        super().close()
