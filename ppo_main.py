from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import torch
cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
policy_base_dir = os.path.join(root, "model_free", "policy_base")
from common.rl_utils import evaluate_policy, evaluate_policy_ppo_continuous
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# algo_name = 'ppo_discrete'
algo_name = 'ppo_continuous'

if algo_name == 'ppo_discrete':
    root1 = os.path.abspath(os.path.join(policy_base_dir, "ppo_discrete"))
    sys.path.append(root1)
    from ppo_discrete_config import Config
    from ppo_discrete_agent import Agent, SampleData
    # env_name = "CartPole-v1"
    env_name = 'LunarLander-v3'
    seed = 209
elif algo_name == 'ppo_continuous':
    root1 = os.path.abspath(os.path.join(policy_base_dir, "ppo_continuous"))
    sys.path.append(root1)
    from ppo_continuous_config import Config
    from ppo_continuous_agent import Agent, SampleData
    # env_name = 'Pendulum-v1'
    env_name = 'HalfCheetah-v5'
    seed = 0


env = gym.make(env_name)
eval_env = gym.make(env_name)
# 设置随机数种子,提升训练的可复现性
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if env_name == 'Pendulum-v1' or 'HalfCheetah-v5':
    max_action = float(env.action_space.high[0])

agent = Agent(env)

logdir = f"runs/{algo_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)

traj_len, total_steps = 0, 0
Long_Traj = []
while total_steps < Config.Max_train_steps:
    episode_return = 0
    state, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
    env_seed += 1
    done = False

    '''Interact & trian'''
    while not done:
        '''Interact with Env'''
        action, logprob_a = agent.take_action(state) # use stochastic when training
        if algo_name == 'ppo_continuous':
            env_action = 2 * (action - 0.5) * max_action
        next_state, reward, dw, truncated, _ = env.step(env_action) # dw: dead&win; tr: truncated
        if env_name == 'Pendulum-v1': reward = (reward + 8) / 8
        if env_name == 'LunarLander-v3' and reward <= -100: reward = -30  # good for LunarLander
        done = (dw or truncated)
        episode_return += reward

        '''Store the current transition'''
        sample = SampleData(state, action, reward, next_state, logprob_a, dw, done)
        Long_Traj.append(sample)
        state = next_state

        traj_len += 1
        total_steps += 1

        '''Update if its time'''
        if traj_len % Config.max_traj_len == 0:
            actor_loss, critic_loss = agent.update(Long_Traj)
            Long_Traj = []
            traj_len = 0
            writer.add_scalar('actor_loss', actor_loss, global_step=total_steps)
            writer.add_scalar('critic_loss', critic_loss, global_step=total_steps)

        '''Eval & Record'''
        if total_steps % Config.eval_interval == 0:
            if algo_name == 'ppo_discrete':
                score = evaluate_policy(eval_env, agent, turns=10) # evaluate the policy for 3 times, and get averaged result
            elif algo_name == 'ppo_continuous':
                score = evaluate_policy_ppo_continuous(eval_env, agent, turns=10)
            writer.add_scalar('ep_r', score, global_step=total_steps)


env = gym.make(env_name,render_mode = 'human')
obs, _ = env.reset()
action = agent.best_action(obs)    
episode_over = False
while not episode_over:
    obs, r, terminated, truncated, _ = env.step(action)
    next_action = agent.best_action(obs)
    action = next_action
    episode_over = terminated or truncated
env.close()