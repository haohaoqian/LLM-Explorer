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
import re
import pickle
import random
from openai import OpenAI


import warnings
warnings.filterwarnings("ignore")



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='Pong-v5', help='name of the environment to run')
parser.add_argument('--manual_seed', type=int, default=1, help='manual seed for reproducibility')
parser.add_argument('--LLM_name', type=str, default='gpt-4o-2024-08-06', help='name of the LLM')
parser.add_argument('--LLM_temperature', type=float, default=0.0, help='temperature of the LLM')
parser.add_argument('--adjust_frequency', type=int, default=1, help='adjust frequency of the LLM')
parser.add_argument('--sample_rate', type=int, default=100, help='sample rate of the LLM')
parser.add_argument('--prompt_type', type=str, default='full', help='type of prompt to use')
parser.add_argument('--episodes', type=int, default=500, help='number of episodes to run')
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
parser.add_argument('--LLM_max_try', type=int, default=10, help='number of tries to get the LLM output')
args = parser.parse_args()

env_name = args.env_name
manual_seed = args.manual_seed

LLM_name = args.LLM_name
LLM_temperature = args.LLM_temperature
adjust_frequency = args.adjust_frequency
sample_rate = args.sample_rate
prompt_type = args.prompt_type

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
LLM_max_try = args.LLM_max_try

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(manual_seed)
if device=='cuda':
    torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)
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
        self.model_dir=f'DQN_{LLM_name}_adjust{adjust_frequency}_sample{sample_rate}_{prompt_type}_temperature{LLM_temperature}_seed{manual_seed}_{self.time_data}'
        os.makedirs(os.path.join(self.game_dir,self.model_dir),exist_ok=True)

        os.system('cp DQN_LLM.py '+os.path.join(self.game_dir,self.model_dir,'DQN_LLM.py'))
        with open(os.path.join(self.game_dir,self.model_dir,'args.pkl'), 'wb') as f:
            pickle.dump(args, f)
            
    def _reset(self):
        self.state, _ = self.env.reset()
        self.timestep = 0
        self.total_reward = 0      

    def select_action(self):
        if np.random.random() < self.epsilon:
            if probs_llm != None and len(probs_llm)==action_dim:
                actions = [q for q in range(0,action_dim)]
                action = random.choices(actions, weights=probs_llm, k=1)[0]
            else:
                action = self.env.action_space.sample() # sample
            action_flag = 1
        else:
            state = np.array([self.state])
            state = torch.tensor(state).to(self.device)
            action = np.argmax(self.model(state).cpu().detach().numpy())
            action_flag = 0

        return action, action_flag    

    def get_experience(self,if_render):
        episode_reward = None
        action, action_flag = self.select_action()
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
print(action_dim)

Experience = collections.namedtuple('Experience',field_names=['state', 'action', 'reward', 'done', 'next_state'])
buffer = ExperienceReplay(buffer_size)
q_net = DQN().to(device)
q_target = copy.deepcopy(q_net).to(device)
epsilon = start_epsilon
episode_rewards = []
agent = Agent()
optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)

description = None
probs_llm = None

token_input_stage1 = []
token_output_stage1 = []
token_input_stage2 = []
token_output_stage2 = []

if 'Llama' in LLM_name:
    LLM_name = 'meta-llama/Meta-'+LLM_name

step_count = 0
for episode in tqdm(range(episodes)):
    if step_count >= max_step:
        break

    terminate = False
    if_render = False

    if episode % save_interval == 0:
        if_render = True

    if episode != 0 and episode % adjust_frequency == 0:
        system_prompt1 = 'You are describing the last episode of the training process on a task. ' + env_describe[env_name]
        base_prompt1 = 'In the last episode, the reward is [reward], and the action sequence extracted at intervals is ([action]). Please analyze the data, generate a description, and provide possible strategy recommendations. '

        step = len(action_record) // sample_rate
        action_gpt_sample = action_record[len(action_record) % sample_rate::step]
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
                probs_llm = None

                output_tokens_stage1=0
                prompt_tokens_stage1=0
                total_tokens_stage1=0
                time.sleep(1)
                print('API call failed, retrying...')
                continue

            if description!=None and len(description)>0:
                break
            else:
                description = None
                probs_llm = None

                output_tokens_stage1=0
                prompt_tokens_stage1=0
                total_tokens_stage1=0

        if description != None:
            system_prompt2 = 'You are determining the probability distribution for exploration actions in reinforcement learning. ' + env_describe[env_name]
            base_prompt2 = f'Here is a description of the situation in the previous episode: [description]. Based on the above information, please analyze what kind of actions should be selected to better improve the task effectiveness. Please output the distribution of the {action_dim} action explorations for the next episode based on your analysis in decimal form. Your output format should be: {{'
            for i in range(action_dim):
                base_prompt2 += f'{i}: [probability]'
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
                    probs_llm = None
                    output_tokens_stage2=0
                    prompt_tokens_stage2=0
                    total_tokens_stage2=0

                    time.sleep(1)
                    print('API call failed, retrying...')
                    continue

                pattern = r'\b\d*\.\d+\b'
                matches = re.findall(pattern, ans)
                probs = matches[-action_dim:]
                probs_llm = [float(item) for item in probs]
                
                if probs_llm!=None and len(probs_llm)==action_dim:
                    break
                else:
                    ans = None
                    probs_llm = None

                    output_tokens_stage2=0
                    prompt_tokens_stage2=0
                    total_tokens_stage2=0

        else:
            base_prompt2 = 'None'
            ans = None
            probs_llm = None
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
                       'Stage1-Output:'+str(description)+'\n-----------------------\n\n'+\
                       'Stage2-Input:'+base_prompt2+'\n-----------------------\n\n'+\
                       'Stage2-Output:'+str(ans)+'\n\n\n')
        
        with open(os.path.join(agent.game_dir,agent.model_dir, 'llm_probs.txt'), 'a') as file:
            file.write(f'episide:{episode} ' + str(probs_llm) + '\n')

    action_record = []
    action_flag_record = []
    reward_record = []

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