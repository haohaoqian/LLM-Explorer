import os
from collections import deque

import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import re
import cv2
import time
import torch
import pickle

import sys
sys.path.append('..')

from openai import OpenAI


# Pre-defined library
from agent import Agent
from memory import ReplayMemory



parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='Pong-v5', help='name of the environment to run')
parser.add_argument('--manual_seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--episodes', type=int, default=800, help='number of episodes to run')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--max_step', type=int, default=310000, help='maximum number of steps to run')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
# parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model 1 (state dict)')
parser.add_argument('--model2', type=str, metavar='PARAMS', help='Pretrained model 2 (state dict)')
parser.add_argument('--memory-capacity', type=int, default=10000, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=32, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=1000, metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=256, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=10000, metavar='STEPS', help='Number of steps before starting training')
### TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=50000, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--cpu_num', type=int, default=1, help='number of CPU cores to use')

parser.add_argument('--start_epsilon', type=float, default=1, help='starting value of epsilon')
parser.add_argument('--min_epsilon', type=float, default=0.1, help='minimum value of epsilon')
parser.add_argument('--epsilon_decay', type=float, default=0.99999, help='decay rate of epsilon')
parser.add_argument('--save_folder', type=str, default='model', help='folder to save the model')

parser.add_argument('--LLM_name', type=str, default='gpt-4o-2024-08-06', help='name of the LLM')
parser.add_argument('--LLM_temperature', type=float, default=0.0, help='temperature of the LLM')
parser.add_argument('--adjust_frequency', type=int, default=1, help='adjust frequency of the LLM')
parser.add_argument('--sample_rate', type=int, default=100, help='sample rate of the LLM')
parser.add_argument('--prompt_type', type=str, default='full', help='type of prompt to use')
parser.add_argument('--LLM_max_try', type=int, default=10, help='number of tries to get the LLM output')
args = parser.parse_args()

class Env_Wrapper(gym.Wrapper):
    def __init__(self, env, args):
        super(Env_Wrapper, self).__init__(env)
        # Set up parameters 
        self.window       = args.history_length             # Number of frames to save in the history
        self.state_buffer =  deque([], maxlen=self.window)  # Buffer to store historical frames 
        self.device       = args.device                     # Running on either 'cpu' or 'cuda'
        self.env          = env                             # Refrence to the original environment 
        
    def _reset_buffer(self):
        # Reset the state buffer with zero tensors 
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(130, 144, device=self.device))
    
    def process_state(self, state):
        # Preprocess the state by converting to grayscale
        gray = cv2.cvtColor(state[28:-52,8:152], cv2.COLOR_RGB2GRAY)
        return torch.tensor(gray, device = self.device)
        
    def reset(self):
        # Reset the state buffer and obtain the initial satet from the environment 
        self._reset_buffer()
        raw_state       = self.env.reset()  #
        processed_state = self.process_state(raw_state[0])
        
        # Update the state buffer with the processed state
        self.state_buffer.append(processed_state)
        
        # return the raw state and the stacked state buffer where each being normalize to [0,1]
        return raw_state, torch.stack(list(self.state_buffer) , 0) / 255
    
    def step(self, action):
        # Select an action in the environment and obtain the next state
        next_state, reward, done, truncated, info = self.env.step(action)
        
        # preprocess the next state
        processed_state = self.process_state(next_state)
        
        # Update the state buffer with the preprocessed state
        self.state_buffer.append(processed_state)
        
        # Return (next state, normalized stacked state buffer, reward, end_status, info) 
        return next_state, torch.stack(list(self.state_buffer), 0) / 255, reward, done, info

def render(x, step, reward):
    plt.figure(figsize=(6, 6))
    plt.clf()
    plt.axis("off")

    plt.title(f"step: {step}, r={reward}")
    plt.imshow(x, cmap=plt.cm.gray)
    plt.pause(0.0001)   # pause for plots to update


# Get the arguments and create the directory if it not exist
manual_seed = args.manual_seed
print(f"Manual Seed: {manual_seed}")
env_name    = args.env_name
print(f"Environment: {env_name}")

LLM_name = args.LLM_name
LLM_temperature = args.LLM_temperature
adjust_frequency = args.adjust_frequency
sample_rate = args.sample_rate
prompt_type = args.prompt_type
LLM_max_try = args.LLM_max_try

save_folder = args.save_folder
game_dir=os.path.join('..',save_folder,env_name)
time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
model_dir=f'Rainbow_DQN_{LLM_name}_adjust{adjust_frequency}_sample{sample_rate}_{prompt_type}_temperature{LLM_temperature}_seed{manual_seed}_{time_data}'
os.makedirs(os.path.join(game_dir,model_dir),exist_ok=True)
with open(os.path.join(game_dir,model_dir,'args.pkl'), 'wb') as f:
    pickle.dump(args, f)

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

if 'gpt' in LLM_name:
    API_KEY = YOUR_API_KEY
    client = OpenAI(api_key=API_KEY,)

elif 'Llama' in LLM_name:
    API_KEY = YOUR_API_KEY
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepinfra.com/v1/openai",)

if prompt_type == 'full':
    from prompt import env_describe_full as env_describe
elif prompt_type == 'name':
    from prompt import env_describe_name as env_describe

# Print out each settings in the arguments
print("Parameter Settings")
for k, v in vars(args).items():
    print( f'{k:<20} : {str(v)} ')  # Print each parameter along with its value

# Select the device used to train the neural network
if torch.cuda.is_available() and not args.disable_cuda:
    # If CUDA (GPU) is available and not explicitly disabled in arguments
    args.device = torch.device('cuda') # Set the device to CUDA
    torch.cuda.manual_seed(np.random.randint(1, 10000)) # Set CUDA random seed
    torch.backends.cudnn.enabled = args.enable_cudnn # Enable cuDNN if specified
else:
    # If CUDA is not available or explicitly disabled
    args.device = torch.device('cpu') # Set the device to CPU

# Set random seed
torch.manual_seed(manual_seed)
if args.device==torch.device('cuda'):
    torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
print('manual_seed=', manual_seed)
  
print(f"Device used: {args.device}") # Print the selected device


env              = Env_Wrapper(gym.make(f'ALE/{env_name}'), args) # Create the environment
raw_obs, observe = env.reset()                           # Initial observation
dqn              = Agent(args, env, manual_seed)                        # Create Agent 
mem              = ReplayMemory(args, args.memory_capacity)# Create the memory buffer

action_dim = env.action_space.n
print(action_dim)

episodes = args.episodes
max_step = args.max_step
priority_weight_increase = (1 - args.priority_weight) / (args.max_step - args.learn_start) # Priority Weight
min_epsilon             = args.min_epsilon
epsilon_decay           = args.epsilon_decay

# 4. Train the Agent
dqn.train()       # Set the deep learning model as train() mode

step_count = 0
raw_obs, state = env.reset()
accu_reward    =  0

description = None
probs_llm = None

episode_rewards = []
token_input_stage1 = []
token_output_stage1 = []
token_input_stage2 = []
token_output_stage2 = []

for episode in tqdm(range(episodes)):
    if step_count >= max_step:
        break

    if episode != 0 and episode % adjust_frequency == 0:
        system_prompt1 = 'You are describing the last episode of the training process on a task. ' + env_describe[env_name]
        base_prompt1 = 'In the last episode, the reward is [reward], and the action sequence extracted at intervals is ([action]). Please analyze the data, generate a description, and provide possible strategy recommendations. '

        step = len(action_record) // sample_rate
        action_gpt_sample = action_record[len(action_record) % sample_rate::step]
        base_prompt1 = base_prompt1.replace('[reward]', str(accu_reward)) 
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

        np.save(os.path.join(game_dir,model_dir,'token_input_stage1.npy'),np.array(token_input_stage1))
        np.save(os.path.join(game_dir,model_dir,'token_output_stage1.npy'),np.array(token_output_stage1))
        np.save(os.path.join(game_dir,model_dir,'token_input_stage2.npy'),np.array(token_input_stage2))
        np.save(os.path.join(game_dir,model_dir,'token_output_stage2.npy'),np.array(token_output_stage2))

        with open(os.path.join(game_dir,model_dir, f'llm_text.txt'), 'a') as file:
            file.write(f'=====================Episode{episode}====================\n'+\
                       'Stage1-Input:'+base_prompt1+'\n-----------------------\n\n'+\
                       'Stage1-Output:'+str(description)+'\n-----------------------\n\n'+\
                       'Stage2-Input:'+base_prompt2+'\n-----------------------\n\n'+\
                       'Stage2-Output:'+str(ans)+'\n\n\n')
        
        with open(os.path.join(game_dir,model_dir, 'llm_probs.txt'), 'a') as file:
            file.write(f'episide:{episode} ' + str(probs_llm) + '\n')

    done = False
    action_record = []
    action_flag_record = []
    reward_record = []

    while not done:
        torch.cuda.empty_cache()

        if step_count % args.replay_frequency == 0:  
            dqn.reset_noise()              # Draw a new set of noisy weights
        
        action, action_flag = dqn.act(state,probs_llm)
        raw_obs, next_state, reward, done, info = env.step(action) # Perform the action

        # reward = max(min(reward, args.reward_clip), -args.reward_clip) # Clip reward
        
        accu_reward += reward           
        mem.append(state[-1], action, reward, done)  # Append transition to memory for left paddle view

        # Train and test after accumulate enough sample from the episode
        if step_count >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            # Train the model with n-step distributional double-Q learning
            if step_count % args.replay_frequency == 0:
                dqn.learn(mem)  

            # Update target network (DONE)
            if step_count % args.target_update == 0: 
                dqn.update_target_net()

            # Checkpoint the network (DONE)
            if (args.checkpoint_interval != 0) and (step_count % args.checkpoint_interval == 0):
                torch.save(dqn.online_net.state_dict(), os.path.join(game_dir,model_dir,f'Rainbow_DQN_{episode}.pth'))

        state = next_state

        dqn.epsilon = max(dqn.epsilon*epsilon_decay,min_epsilon)
        step_count += 1

        action_record.append(action)
        action_flag_record.append(action_flag)
        reward_record.append(reward)

        if done:   # Reset the env if it is done
            raw_obs, state = env.reset()

            episode_rewards.append(accu_reward)
            mean_reward = np.mean(episode_rewards[-100:])
            tqdm.write(f"episode {episode}, episode reward: {accu_reward}, mean reward: {mean_reward:.3f}")

            np.save(os.path.join(game_dir,model_dir,'episode_rewards.npy'),np.array(episode_rewards))
            np.save(os.path.join(game_dir,model_dir,f'action_record-episode{episode}.npy'),np.array(action_record))
            np.save(os.path.join(game_dir,model_dir,f'action_flag_record-episode{episode}.npy'),np.array(action_flag_record))
            np.save(os.path.join(game_dir,model_dir,f'reward_record-episode{episode}.npy'),np.array(reward_record))

            accu_reward    =  0