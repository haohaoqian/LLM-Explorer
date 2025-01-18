# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle
import time

import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, FrameStack, ResizeObservation, TransformObservation
import numpy as np
import torch
from tqdm import trange,tqdm

from agent import Agent
from memory import ReplayMemory



# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='ms_pacman', help='ATARI game')
parser.add_argument('--max_step', type=int, default=310000, help='maximum number of steps to run')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--episodes', type=int, default=500, help='number of episodes to run')
parser.add_argument('--architecture', type=str, default='data-efficient', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=1, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=20, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(2e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(1600), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')

parser.add_argument('--cpu_num', type=int, default=1, help='number of CPU cores to use')

parser.add_argument('--start_epsilon', type=float, default=0.5, help='starting value of epsilon')
parser.add_argument('--min_epsilon', type=float, default=0.1, help='minimum value of epsilon')
parser.add_argument('--epsilon_decay', type=float, default=0.99999, help='decay rate of epsilon')
parser.add_argument('--save_folder', type=str, default='model', help='folder to save the model')

# Setup
args = parser.parse_args()

cpu_num = args.cpu_num
def set_cpu_num(cpu_num):
    if cpu_num <= 0: return
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
set_cpu_num(cpu_num)

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

manual_seed = args.seed
print(f"Manual Seed: {manual_seed}")
env_name    = args.game
print(f"Environment: {env_name}")
xid = 'curl-' + args.game + '-' + str(manual_seed)
args.id = xid

save_folder = args.save_folder
game_dir=os.path.join('..',save_folder,env_name)
time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
model_dir=f'CURL_seed{manual_seed}_{time_data}'
os.makedirs(os.path.join(game_dir,model_dir),exist_ok=True)
with open(os.path.join(game_dir,model_dir,'args.pkl'), 'wb') as f:
    pickle.dump(args, f)


metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')


torch.manual_seed(manual_seed)
if args.device==torch.device('cuda'):
    torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
print('manual_seed=', manual_seed)


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)


# Environment
env = FrameStack(TransformObservation(ResizeObservation(GrayScaleObservation(gym.make(f'ALE/{env_name}', render_mode='rgb_array', full_action_space=False)), (110,84)), lambda x: np.array(x[18:102,:]).astype(np.float32) / 255.0), 4)
env.seed(manual_seed)
env.action_space.seed(manual_seed)

action_dim = env.action_space.n

# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

  mem = load_memory(args.memory, args.disable_bzip_memory)

else:
  mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.max_step - args.learn_start)


# # Construct validation memory
# val_mem = ReplayMemory(args, args.evaluation_size)
# T, done = 0, True
# while T < args.evaluation_size:
#   if done:
#     state, done = env.reset(), False

#   next_state, _, done = env.step(np.random.randint(0, action_space))
#   val_mem.append(state, None, None, done)
#   state = next_state
#   T += 1

# if args.evaluate:
#   dqn.eval()  # Set DQN (online network) to evaluation mode
#   avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
#   print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
# else:

# Training loop
dqn.train()
T = 0
state, _ = env.reset()
done = False

accu_reward = 0
episodes_count = 0
episode_rewards = []

action_record = []
action_flag_record = []
reward_record = []

max_episode = args.episodes
min_epsilon             = args.min_epsilon
epsilon_decay           = args.epsilon_decay

for T in trange(1, args.max_step + 1):
  if done:
    state, _ = env.reset()
    done = False

    episode_rewards.append(accu_reward)
    mean_reward = np.mean(episode_rewards[-100:])
    tqdm.write(f"episode {episodes_count}, episode reward: {accu_reward}, mean reward: {mean_reward:.3f}")

    np.save(os.path.join(game_dir,model_dir,'episode_rewards.npy'),np.array(episode_rewards))
    np.save(os.path.join(game_dir,model_dir,f'action_record-episode{episodes_count}.npy'),np.array(action_record))
    np.save(os.path.join(game_dir,model_dir,f'action_flag_record-episode{episodes_count}.npy'),np.array(action_flag_record))
    np.save(os.path.join(game_dir,model_dir,f'reward_record-episode{episodes_count}.npy'),np.array(reward_record))

    action_record = []
    action_flag_record = []
    reward_record = []

    accu_reward = 0
    episodes_count += 1

    if episodes_count >= max_episode:
      break

  if T % args.replay_frequency == 0:
    dqn.reset_noise()  # Draw a new set of noisy weights

  state = np.array([state])
  state = torch.tensor(state).to(args.device)

  action,action_flag = dqn.act(state)  # Choose an action greedily (with noisy weights)
  next_state, reward, done, _, _ = env.step(action)  # Step
  accu_reward += reward

  action_record.append(action)
  action_flag_record.append(action_flag)
  reward_record.append(reward)

  if args.reward_clip > 0:
    reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
  mem.append(state, action, reward, done)  # Append transition to memory

  # Train and test
  if T >= args.learn_start:
    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

    if T % args.replay_frequency == 0:
      #for _ in range(4):
      dqn.learn(mem)  # Train with n-step distributional double-Q learning
      dqn.update_momentum_net() # MoCo momentum upate

    # if T % args.evaluation_interval == 0:
    #   dqn.eval()  # Set DQN (online network) to evaluation mode
    #   avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)  # Test
    #   log('T = ' + str(T) + ' / ' + str(args.max_step) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
    #   dqn.train()  # Set DQN (online network) back to training mode

    #   # If memory path provided, save it
    #   if args.memory is not None:
    #     save_memory(mem, args.memory, args.disable_bzip_memory)

    # Update target network
    if T % args.target_update == 0:
      dqn.update_target_net()

    # Checkpoint the network
    if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
      torch.save(dqn.online_net.state_dict(), os.path.join(game_dir,model_dir,f'CURL_{episodes_count}.pth'))

  dqn.epsilon = max(dqn.epsilon*epsilon_decay,min_epsilon)
  state = next_state

env.close()