import gymnasium as gym
import numpy as np
import collections
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, FrameStack, ResizeObservation, TransformObservation
from tqdm import tqdm
from torch import optim
import imageio
import os
import time
import pickle

import warnings
warnings.filterwarnings("ignore")



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='Pong-v5', help='name of the environment to run')
parser.add_argument('--manual_seed', type=int, default=1, help='manual seed for reproducibility')
parser.add_argument('--episodes', type=int, default=800, help='number of episodes to run')
parser.add_argument('--max_step', type=int, default=310000, help='maximum number of steps to run')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('--start_epsilon', type=float, default=1, help='starting value of epsilon')
parser.add_argument('--min_epsilon', type=float, default=0.1, help='minimum value of epsilon')
parser.add_argument('--epsilon_decay', type=float, default=0.99999, help='decay rate of epsilon')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--buffer_size', type=int, default=10000, help='minimum length of the buffer')
parser.add_argument('--target_update_interval', type=int, default=1000, help='interval to update the target network')
parser.add_argument('--save_interval', type=int, default=50, help='interval to save the model')
parser.add_argument('--save_folder', type=str, default='model', help='folder to save the model')
parser.add_argument('--cpu_num', type=int, default=1, help='number of CPU cores to use')
args = parser.parse_args()

env_name = args.env_name
manual_seed = args.manual_seed

episodes = args.episodes
max_step = args.max_step
batch_size = args.batch_size
start_epsilon = args.start_epsilon
min_epsilon = args.min_epsilon
epsilon_decay = args.epsilon_decay
gamma = args.gamma
learning_rate = args.learning_rate
buffer_size = args.buffer_size
target_update_interval = args.target_update_interval
save_interval = args.save_interval
save_folder = args.save_folder

cpu_num = args.cpu_num


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(manual_seed)
if device=='cuda':
    torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
print('manual_seed=', manual_seed)

def set_cpu_num(cpu_num):
    if cpu_num <= 0: return
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
set_cpu_num(cpu_num)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(4,4*8,8,stride=4,device=device)
        self.Conv2 = nn.Conv2d(4*8,4*8*2,4,stride=2,device=device)
        self.Conv3 = nn.Conv2d(64,64,3,stride=1,device=device)
        self.Linear1 = nn.Linear(3136,512)   
        self.Linear2 = nn.Linear(512,action_dim)    
        
    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv3(x))
        x = torch.flatten(x,1,3)
        x = F.relu(self.Linear1(x))
        x = self.Linear2(x)
        return x

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
  
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer),batch_size,replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions,dtype=np.int64), np.array(rewards,dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)
    
class Agent:
    def __init__(self, mode="training"):
        self.env = env
        self.epsilon = start_epsilon
        self.device = device
        self.buffer = buffer
        self.model = q_net
        self.target_model = q_target
        self.mode = mode
        self.episode = 0
        self.learns = 0
        self.frames = []

        self._reset()

        self.game_dir=os.path.join('..',save_folder,env_name)
        self.time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.model_dir=f'DQN_seed{manual_seed}_{self.time_data}'
        os.makedirs(os.path.join(self.game_dir,self.model_dir),exist_ok=True)

        os.system('cp DQN.py '+os.path.join(self.game_dir,self.model_dir,'DQN.py'))
        with open(os.path.join(self.game_dir,self.model_dir,'args.pkl'), 'wb') as f:
            pickle.dump(args, f)
            
    def _reset(self):
        self.state, _ = self.env.reset()
        self.timestep = 0
        self.total_reward = 0      

    def select_action(self):
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample() # sample
            action_flag = 1
        else:
            state = np.array([self.state])
            state = torch.tensor(state).to(self.device)
            action = np.argmax(self.model(state).cpu().detach().numpy())
            action_flag = 0

        return action,action_flag

    def get_experience(self,if_render):
        episode_reward = None
        action,action_flag = self.select_action()
        next_state, reward, terminate, _, _ = self.env.step(action)
        
        # Save the current frame as an image
        if if_render:
            frame = self.env.render()  # No need to pass 'mode' parameter
            self.frames.append(frame)
        
        exp = Experience(self.state,action,reward,terminate,next_state)
        self.buffer.append(exp)
        self.state = next_state
        self.timestep += 1
        self.total_reward += reward
        
        if terminate:
            episode_reward = self.total_reward
            tqdm.write(f"timestep {self.timestep} Score: {episode_reward}")
            self.episode += 1
            self._reset()
            return True, reward, episode_reward, action, action_flag
        
        if len(agent.buffer) >= buffer_size:
            self.update_weights() 

        return False, reward, episode_reward, action, action_flag
    
    def update_weights(self):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = batch
                
        states_t = torch.tensor(states).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)
        action_values = self.model(states_t).gather(1,actions_t.unsqueeze(-1)).squeeze(-1)
        next_action_values = self.target_model(next_states_t).max(1)[0]
        next_action_values[done_mask] = 0.0
        next_action_values = next_action_values.detach()
        
        expected_action_values = rewards_t + next_action_values*gamma 
        loss_t = nn.MSELoss()(action_values, expected_action_values)
        
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        self.learns += 1
            
        if self.learns % target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            tqdm.write(f"episode {self.episode}: target model weights updated")

    def save(self,episode_count):
        torch.save(self.model.state_dict(),os.path.join(self.game_dir,self.model_dir,f'model-episode{episode_count}.pth'))


env = FrameStack(TransformObservation(ResizeObservation(GrayScaleObservation(gym.make(f'ALE/{env_name}', render_mode='rgb_array', full_action_space=False)), (110,84)), lambda x: np.array(x[18:102,:]).astype(np.float32) / 255.0), 4)
env.seed(manual_seed)
env.action_space.seed(manual_seed)

action_dim = env.action_space.n

Experience = collections.namedtuple('Experience',field_names=['state', 'action', 'reward', 'done', 'next_state'])
buffer = ExperienceReplay(buffer_size)
q_net = DQN().to(device)
q_target = copy.deepcopy(q_net).to(device)
epsilon = start_epsilon
episode_rewards = []
agent = Agent()
optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)

step_count = 0
for episode in tqdm(range(episodes)):
    if step_count >= max_step:
        break
    
    action_record = []
    action_flag_record = []
    reward_record = []

    terminate = False
    if_render = False
    if episode % save_interval ==0:
        if_render = True

    while not terminate:
        step_count += 1
        agent.epsilon = max(agent.epsilon*epsilon_decay,min_epsilon)
        terminate, reward, episode_reward, action, action_flag = agent.get_experience(if_render)

        action_record.append(action)
        action_flag_record.append(action_flag)
        reward_record.append(reward)
            
        if terminate:
            episode_rewards.append(episode_reward)

            mean_reward = np.mean(episode_rewards[-100:])
            tqdm.write(f"episode {episode}, episode reward: {episode_reward}, mean reward: {mean_reward:.3f}")

            np.save(os.path.join(agent.game_dir,agent.model_dir,'episode_rewards.npy'),np.array(episode_rewards))
            np.save(os.path.join(agent.game_dir,agent.model_dir,f'action_record-episode{episode}.npy'),np.array(action_record))
            np.save(os.path.join(agent.game_dir,agent.model_dir,f'action_flag_record-episode{episode}.npy'),np.array(action_flag_record))
            np.save(os.path.join(agent.game_dir,agent.model_dir,f'reward_record-episode{episode}.npy'),np.array(reward_record))

    if (episode + 1) % save_interval ==0:
        agent.save(episode_count=episode)
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(episode_rewards)), episode_rewards)
        ax.set(xlabel='timestep', ylabel='episode reward',title="reward over time")
        ax.grid()
        fig.savefig(os.path.join(agent.game_dir,agent.model_dir,f"result.png"))
        # Save frames as GIF
        imageio.mimsave(os.path.join(agent.game_dir,agent.model_dir,f'gif-episode{episode}.gif'), agent.frames, fps=30)
        agent.frames = []  # Clear frames for the next episode

env.reset()        
env.close()