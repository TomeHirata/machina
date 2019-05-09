import gym
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import pickle
from pprint import pprint
import os
import json
import argparse
import pybullet_envs
"""
Script for taking movie of learned policy.
"""
from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM, QNet
from machina.utils import measure, set_device
from machina import logger
from machina.samplers import EpiSampler
from machina.envs import GymEnv, C2DEnv
from machina.noise import OUActionNoise
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol, DeterministicActionNoisePol, ArgmaxQfPol
from machina.vfuncs import CEMDeterministicSAVfunc
import machina as mc
"""
Script for taking movie of learned policy.
"""


parser = argparse.ArgumentParser()
parser.add_argument('--pol_dir', type=str, default='garbage',
                    help='Directory path storing file of optimal policy model.')
parser.add_argument('--pol_fname', type=str, default='pol_max.pkl',
                    help='File name of optimal policy model.')
parser.add_argument('--env_name', type=str,
                    default='Pendulum-v0', help='Name of environment.')
parser.add_argument('--c2d', action='store_true',
                    default=False, help='If True, action is discretized.')
parser.add_argument('--record', action='store_true',
                    default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')
parser.add_argument('--rnn', action='store_true',
                    default=False, help='If True, network is reccurent.')
parser.add_argument('--pol_h1', type=int, default=200,
                    help='Hidden size of layer1 of policy.')
parser.add_argument('--pol_h2', type=int, default=100,
                    help='Hidden size of layer2 of policy.')

parser.add_argument('--num_epis', type=int, default=5,
                    help='Number of episodes of expert trajectories.')
parser.add_argument('--ddpg', action='store_true',
                    default=False, help='If True, policy for DDPG is used.')
parser.add_argument('--num_iter', type=int, default=2,
                    help='Number of iteration of CEM.')
parser.add_argument('--num_sampling', type=int, default=60,
                    help='Number of samples sampled from Gaussian in CEM.')
parser.add_argument('--num_best_sampling', type=int, default=6,
                    help='Number of best samples used for fitting Gaussian in CEM.')
parser.add_argument('--eps', type=float, default=0.2,
                    help='Probability of random action in epsilon-greedy policy.')
parser.add_argument('--save_memory', action='store_true', default=False,
                    help='If true, save memory while need more computation time by for-sentence.')
parser.add_argument('--cem', action='store_true',
                    default=False, help='If True, policy for cross entropy method is used.')
args = parser.parse_args()

if not os.path.exists(args.pol_dir):
    os.mkdir(args.pol_dir)

with open(os.path.join(args.pol_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.pol_dir, 'optimal_movie'), record_video=True, video_schedule=lambda x: True)
env.env.seed(args.seed)
if args.c2d:
    env = C2DEnv(env)

observation_space = env.observation_space
action_space = env.action_space

if args.ddpg:
    pol_net = PolNet(observation_space, action_space,
                     args.h1, args.h2, deterministic=True)
    noise = OUActionNoise(action_space.shape)
    pol = DeterministicActionNoisePol(
        observation_space, action_space, pol_net, noise)
elif args.cem:
    qf_net = QNet(observation_space, action_space, args.pol_h1, args.pol_h2)
    qf = CEMDeterministicSAVfunc(
    observation_space, action_space, qf_net,
    num_sampling=args.num_sampling,
    num_best_sampling=args.num_best_sampling,
    num_iter=args.num_iter,
    save_memory=args.save_memory)
    pol = ArgmaxQfPol(env.observation_space, env.action_space, qf, eps=args.eps)
else:
    if args.rnn:
        pol_net = PolNetLSTM(observation_space, action_space,
                             h_size=256, cell_size=256)
    else:
        pol_net = PolNet(observation_space, action_space)
    if isinstance(action_space, gym.spaces.Box):
        pol = GaussianPol(observation_space, action_space, pol_net, args.rnn)
    elif isinstance(action_space, gym.spaces.Discrete):
        pol = CategoricalPol(
            observation_space, action_space, pol_net, args.rnn)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        pol = MultiCategoricalPol(
            observation_space, action_space, pol_net, args.rnn)
    else:
        raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')


sampler = EpiSampler(env, pol, num_parallel=1,  seed=args.seed)

with open(os.path.join(args.pol_dir, 'models', args.pol_fname), 'rb') as f:
    pol.load_state_dict(torch.load(
        f, map_location=lambda storage, location: storage))


epis = sampler.sample(pol, max_epis=args.num_epis)

rewards = [np.sum(epi['rews']) for epi in epis]
mean_rew = np.mean(rewards)
logger.log('score={}'.format(mean_rew))
del sampler
