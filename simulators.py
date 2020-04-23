from collections import defaultdict, namedtuple
import numpy as np
import torch

import gym

from autoassign import autoassign
# from envs.ant_gather import AntGatherEnv
# from envs.point_gather import PointGatherEnv
from memory import Memory, Trajectory
from torch_utils.torch_utils import get_device


# def make_env(env_name, **env_args):
#     if env_name == 'ant_gather':
#         return PointGather(**env_args)
#     elif env_name == 'point_gather':
#         return PointGatherEnv(**env_args)
#     else:
#         raise NotImplementedError


class Simulator:
    @autoassign(exclude=('env_name', 'env_args'))
    def __init__(self, env_name, policy, n_trajectories, trajectory_len, obs_filter=None, **env_args):
        self.env = np.asarray([gym.make(env_name, **env_args) for i in range(n_trajectories)])
        self.n_trajectories = n_trajectories

        for env in self.env:
            env._max_episode_steps = trajectory_len

        self.device = get_device()


class SinglePathSimulator:
    def __init__(self, env_name, policy, n_trajectories, trajectory_len, state_filter=None,
                 **env_args):
        Simulator.__init__(self, env_name, policy, n_trajectories, trajectory_len, state_filter,
                           **env_args)
        self.embeddings= env_args['embeddings']

    def run_sim(self):
        self.policy.eval()

        with torch.no_grad():
            trajectories = np.asarray([Trajectory() for i in range(self.n_trajectories)])
            continue_mask = np.ones(self.n_trajectories)

            for env, trajectory in zip(self.env, trajectories):
                # obs = torch.tensor(env.reset()).float()
                obs = torch.from_numpy(self.embeddings.embed(env.reset())).float().view(-1)

                # Maybe batch this operation later
                if self.obs_filter:
                    obs = self.obs_filter(obs)

                trajectory.observations.append(obs)

            while np.any(continue_mask):
                continue_indices = np.where(continue_mask)
                trajs_to_update = trajectories[continue_indices]
                continuing_envs = self.env[continue_indices]

                policy_input = torch.stack([(trajectory.observations[-1]).clone().detach().to(self.device).view(-1)
                                            for trajectory in trajs_to_update])

                weight_dists = self.policy(policy_input)
                weights = weight_dists.sample().to(self.device)
                item_embeddings = torch.from_numpy(self.embeddings.get_embedding_vector()).float().to(self.device)
                indices = []
                for i in range(0, weights.shape[1],100):
                    weights_ = weights[:,i:i+100]
                    scores = torch.mm(item_embeddings, torch.transpose(weights_,0,1))
                    val, idx = torch.max(scores,0)
                    indices.append(idx)
                
                actions = torch.transpose(torch.stack(indices),0,1).cpu()

                for env, action, trajectory in zip(continuing_envs, actions, trajs_to_update):
                    reward, obs, trajectory.done, cost = env.step(action.numpy())

                    obs = torch.from_numpy(self.embeddings.embed(obs)).float().view(-1)
                    reward = torch.tensor(reward, dtype=torch.float)
                    action = torch.from_numpy(self.embeddings.embed(action)).float().view(-1)
                    cost = torch.tensor(cost, dtype=torch.float)

                    if self.obs_filter:
                        obs = self.obs_filter(obs)

                    trajectory.actions.append(action)
                    trajectory.rewards.append(reward)
                    trajectory.costs.append(cost)

                    if not trajectory.done:
                        trajectory.observations.append(obs)

                continue_mask = np.asarray([1 - trajectory.done for trajectory in trajectories])

        memory = Memory(trajectories)

        return memory
