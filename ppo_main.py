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
from common.rl_utils import evaluate_policy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


algo_name = 'ppo'

if algo_name == 'ppo':
    root1 = os.path.abspath(os.path.join(policy_base_dir, "ppo"))
    sys.path.append(root1)
    from ppo_config import Config
    from ppo_agent import Agent


env_name = "CartPole-v1"
env = gym.make(env_name)
eval_env = gym.make(env_name)
# 设置随机数种子,提升训练的可复现性
seed = 0
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

agent = Agent(env)

logdir = f"runs/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)

return_list = []
total_steps = 0
for i in range(10):
    with tqdm(total=int(Config.num_episodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(Config.num_episodes/10)):
            episode_return = 0
            # Do not use seed directly, or it can overfit to seed
            obs, _ = env.reset(seed=env_seed) 
            env_seed += 1
            done = False
            Episode = []
            while not done:
                if algo_name == 'noisy_dqn' and total_steps < Config.minimal_size: 
                    # steps for random policy to explore 随机探索阶段
                    action = env.action_space.sample()
                else: 
                    action = agent.take_action(obs)
                next_obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replaybuffer.add(obs, action, r, next_obs, done)
                episode_return += r 
                if total_steps % Config.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns = 20)
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                obs = next_obs
                total_steps += 1
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (Config.num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
print(total_steps)


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