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

from openai import OpenAI
import re


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

parser.add_argument('--LLM_name', type=str, default='gpt-4o-2024-08-06', help='name of the LLM')
parser.add_argument('--LLM_temperature', type=float, default=0.0, help='temperature of the LLM')
parser.add_argument('--adjust_frequency', type=int, default=1, help='adjust frequency of the LLM')
parser.add_argument('--sample_rate', type=int, default=100, help='sample rate of the LLM')
parser.add_argument('--prompt_type', type=str, default='full', help='type of prompt to use')
parser.add_argument('--LLM_max_try', type=int, default=10, help='number of tries to get the LLM output')

parser.add_argument('--save_interval', type=int, default=5000, help='interval to save the model')
parser.add_argument('--save_folder', type=str, default='DDPG', help='directory to save the model')
parser.add_argument('--cpu_num', type=int, default=1, help='number of CPU cores to use')
args = parser.parse_args()


env_name = args.env_name
manual_seed = args.manual_seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LLM_name = args.LLM_name
LLM_temperature = args.LLM_temperature
adjust_frequency = args.adjust_frequency
sample_rate = args.sample_rate
prompt_type = args.prompt_type
LLM_max_try = args.LLM_max_try

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

if 'gpt' in LLM_name:
    API_KEY = YOUR_API_KEY
    client = OpenAI(api_key=API_KEY, )

elif 'Llama' in LLM_name:
    API_KEY = YOUR_API_KEY
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepinfra.com/v1/openai", )

if prompt_type == 'full':
    from prompt import env_describe_full as env_describe
elif prompt_type == 'name':
    from prompt import env_describe_name as env_describe

env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

print('state_dim:', state_dim)
print('action_dim:', action_dim)
print('max_action:', max_action)


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
        self.model_dir = f'DDPG_{LLM_name}_adjust{adjust_frequency}_sample{sample_rate}_{prompt_type}_temperature{LLM_temperature}_seed{manual_seed}_{self.time_data}'
        os.makedirs(os.path.join(self.game_dir, self.model_dir), exist_ok=True)

        os.system(f'cp {__file__} ' + os.path.join(self.game_dir, self.model_dir, f'{os.path.basename(__file__)}'))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if bias_llm != None and len(bias_llm)==action_dim:
            noise = np.random.normal(bias_llm, self.epsilon, size=action.shape)
        else:
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

description = None
bias_llm = None

token_input_stage1 = []
token_output_stage1 = []
token_input_stage2 = []
token_output_stage2 = []

for episode in tqdm(range(args.episodes)):
    if episode != 0 and episode % adjust_frequency == 0:
        system_prompt1 = 'You are describing the last episode of the training process on a reinforcement learning task. ' + env_describe[env_name]
        base_prompt1 = f'In the last episode, the reward is [reward], and sequence of the {action_dim}-dimensional action extracted at intervals is ([action]). Please analyze the data, generate a description, and provide possible strategy recommendations.'

        sample_rate_episode = min(sample_rate, len(action_record))
        step = len(action_record) // sample_rate_episode
        action_gpt_sample = action_record[len(action_record) % sample_rate_episode::step]

        base_prompt1 = base_prompt1.replace('[reward]', str(episode_reward)) 
        base_prompt1 = base_prompt1.replace('[action]', ', '.join([str(a) for a in action_gpt_sample]))

        for _ in range(LLM_max_try):
            try:
                completion = client.chat.completions.create(
                model=LLM_name,
                messages=[{"role": "system", "content": system_prompt1},{"role": "user", "content": base_prompt1}],
                temperature=LLM_temperature)
                description = completion.choices[0].message.content

                usage_stage1=completion.usage
                output_tokens_stage1=usage_stage1.completion_tokens
                prompt_tokens_stage1=usage_stage1.prompt_tokens
                total_tokens_stage1=usage_stage1.total_tokens
            except:
                description = None
                bias_llm = None

                output_tokens_stage1=0
                prompt_tokens_stage1=0
                total_tokens_stage1=0
                time.sleep(1)
                # print('API call failed, retrying...')
                continue

            if description!=None and len(description)>0:
                break
            else:
                description = None
                bias_llm = None

                output_tokens_stage1=0
                prompt_tokens_stage1=0
                total_tokens_stage1=0

        if description != None:
            system_prompt2 = 'You are determining the exploration of actions in reinforcement learning. ' + env_describe[env_name]
            base_prompt2 = f'Here is a description of the situation in the previous episode: [description]. Based on the above information, please analyze what kind of actions should be explored to better improve the task effectiveness. The approach is to add a Guassian noise to each dimension of action, and you need to decide the bias of the Guassian noise for each dimension. A bias of 0 means exploring around the origin, a positive bias means exploring more in the positive direction, and a negative bias means exploring more in the negative direction. Please output the bias for each of the {action_dim} dimension of actions for action explorations in the next episode based on your analysis in decimal form. Your output format should be: {{'
            for i in range(action_dim):
                base_prompt2 += f'{i}: [bias]'
                if i!= action_dim-1:
                    base_prompt2+=', '
            base_prompt2+='}.'
            base_prompt2 = base_prompt2.replace('[description]', description)

            for _ in range(LLM_max_try):
                try:
                    completion = client.chat.completions.create(
                    model=LLM_name,
                    messages=[{"role": "system", "content": system_prompt2},{"role": "user", "content": base_prompt2}],
                    temperature=LLM_temperature)
                    ans = completion.choices[0].message.content

                    usage_stage2=completion.usage
                    output_tokens_stage2=usage_stage2.completion_tokens
                    prompt_tokens_stage2=usage_stage2.prompt_tokens
                    total_tokens_stage2=usage_stage2.total_tokens
                except:
                    ans = None
                    bias_llm = None
                    output_tokens_stage2=0
                    prompt_tokens_stage2=0
                    total_tokens_stage2=0

                    time.sleep(1)
                    # print('API call failed, retrying...')
                    continue

                pattern = r'\b\d*\.\d+\b'
                matches = re.findall(pattern, ans)
                bias = matches[-action_dim:]
                bias_llm = [float(item) for item in bias]
                
                if bias_llm!=None and len(bias_llm)==action_dim:
                    break
                else:
                    ans = None
                    bias_llm = None

                    output_tokens_stage2=0
                    prompt_tokens_stage2=0
                    total_tokens_stage2=0

        else:
            base_prompt2 = 'None'
            ans = None
            bias_llm = None
            output_tokens_stage2 = 0
            prompt_tokens_stage2 = 0
            total_tokens_stage2 = 0

        token_input_stage1.append(prompt_tokens_stage1)
        token_output_stage1.append(output_tokens_stage1)
        token_input_stage2.append(prompt_tokens_stage2)
        token_output_stage2.append(output_tokens_stage2)

        np.save(os.path.join(agent.game_dir,agent.model_dir,'token_input_stage1.npy'),np.array(token_input_stage1))
        np.save(os.path.join(agent.game_dir,agent.model_dir,'token_output_stage1.npy'),np.array(token_output_stage1))
        np.save(os.path.join(agent.game_dir,agent.model_dir,'token_input_stage2.npy'),np.array(token_input_stage2))
        np.save(os.path.join(agent.game_dir,agent.model_dir,'token_output_stage2.npy'),np.array(token_output_stage2))

        with open(os.path.join(agent.game_dir,agent.model_dir, f'llm_text.txt'), 'a') as file:
            file.write(f'=====================Episode{episode}====================\n'+\
                       'Stage1-Input:'+base_prompt1+'\n-----------------------\n\n'+\
                       'Stage1-Output:'+description+'\n-----------------------\n\n'+\
                       'Stage2-Input:'+base_prompt2+'\n-----------------------\n\n'+\
                       'Stage2-Output:'+str(ans)+'\n\n\n')
        
        with open(os.path.join(agent.game_dir,agent.model_dir, 'llm_bias.txt'), 'a') as file:
            file.write(f'episide:{episode} ' + str(bias_llm) + '\n')

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