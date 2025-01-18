import argparse
import gymnasium as gym
import numpy as np
import collections
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim

import time
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='Ant-v5', help='name of the environment to run')
parser.add_argument('--manual_seed', type=int, default=1, help='manual seed for reproducibility')
parser.add_argument('--episodes', type=int, default=1500, help='number of episodes to run')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--learning_rate_actor', type=float, default=1e-5, help='learning rate for the actor network')
parser.add_argument('--learning_rate_critic', type=float, default=1e-4, help='learning rate for the critic network')

parser.add_argument('--buffer_size', type=int, default=10000, help='capacity of the replay buffer')
parser.add_argument('--tau', type=float, default=0.005, help='smoothing coefficient for target network updates')
parser.add_argument('--exploration_noise', type=float, default=0.1, help='noise level during exploration')

parser.add_argument('--save_interval', type=int, default=5000, help='interval to save the model')
parser.add_argument('--save_folder', type=str, default='DDPG', help='directory to save the model')
parser.add_argument('--cpu_num', type=int, default=1, help='number of CPU cores to use')
args = parser.parse_args()


env_name = args.env_name
manual_seed = args.manual_seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(manual_seed)
if device == 'cuda':
    torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
print('manual_seed=', manual_seed)

def set_cpu_num(cpu_num):
    if cpu_num <= 0: return
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

set_cpu_num(args.cpu_num)

env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        return self.l3(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate_critic)

        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.num_training = 0
        self.epsilon = args.exploration_noise

        self.game_dir = os.path.join('..', args.save_folder, env_name)
        self.time_data = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
        self.model_dir = f'DDPG_seed{manual_seed}_{self.time_data}'
        os.makedirs(os.path.join(self.game_dir, self.model_dir), exist_ok=True)

        os.system(f'cp {__file__} ' + os.path.join(self.game_dir, self.model_dir, f'{os.path.basename(__file__)}'))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.epsilon, size=action.shape)
        return np.clip(action + noise, -max_action, max_action)

    def update(self):
        if len(self.replay_buffer) < args.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(args.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(device)

        target_Q = self.critic_target(next_states, self.actor_target(next_states))
        target_Q = rewards + ((1 - dones) * args.gamma * target_Q).detach()

        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    def save(self, step_count):
        torch.save(self.actor.state_dict(), os.path.join(self.game_dir, self.model_dir, f'actor_{step_count}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(self.game_dir, self.model_dir, f'critic_{step_count}.pth'))

agent = DDPGAgent(state_dim, action_dim, max_action)
episode_rewards = []
step_counter = 0

args_dict = vars(args)
with open(os.path.join(agent.game_dir,agent.model_dir,'args_config.json'), 'w') as json_file:
    json.dump(args_dict, json_file, indent=4)

for episode in tqdm(range(args.episodes)):
    action_record = []
    reward_record = []

    state, _ = env.reset()
    episode_reward = 0
    timestep = 0 

    while True:  
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.replay_buffer.push((state, action, reward, next_state, float(done)))
        state = next_state
        episode_reward += reward
        
        action_record.append(action)
        reward_record.append(reward)
        agent.update()

        # Increment step counter
        step_counter += 1
        timestep += 1

        if step_counter % args.save_interval == 0:
            agent.save(step_count=f"step_{step_counter}_episode_{episode}")
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(episode_rewards)), episode_rewards)
            ax.set(xlabel='episode', ylabel='episode reward', title="reward over time")
            ax.grid()
            fig.savefig(os.path.join(agent.game_dir,agent.model_dir,f"A_result.png"))
            
        if done or truncated or timestep >= 1000:
            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-100:])
            tqdm.write(f"=={env_name}_DDPG== episode {episode}, episode reward: {episode_reward}, mean reward: {mean_reward:.3f}")
            np.save(os.path.join(agent.game_dir,agent.model_dir,'episode_rewards.npy'),np.array(episode_rewards))
            np.save(os.path.join(agent.game_dir,agent.model_dir,f'action_record-episode{episode}.npy'),np.array(action_record))
            np.save(os.path.join(agent.game_dir,agent.model_dir,f'reward_record-episode{episode}.npy'),np.array(reward_record))
            break


env.close()
agent.save(step_count=f"step_{step_counter}_episode_{args.episodes}")
fig, ax = plt.subplots()
ax.plot(np.arange(len(episode_rewards)), episode_rewards)
ax.set(xlabel='episode', ylabel='episode reward', title="reward over time")
ax.grid()
fig.savefig(os.path.join(agent.game_dir,agent.model_dir,f"A_result.png"))