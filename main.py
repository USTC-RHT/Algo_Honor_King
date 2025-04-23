from collections import defaultdict
import gymnasium as gym
import numpy as np
import os
import sys
cur_dir = os.path.dirname(__file__)
root1 = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo", "model_free", "value_base"))
root2 = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root1)
sys.path.append(root2)

from sarsa_config import Config
from sarsa_agent import Agent
from tqdm import tqdm

# agent = Sarsa(n_row,n_col,epsilon,alpha,gamma,n_action)
env_name = "CliffWalking-v0"    # "FrozenLake-v1"
env = gym.make(env_name)
agent = Agent(env_name)

return_list = []
for i in range(10):
    with tqdm(total=int(Config.num_episodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(Config.num_episodes/10)):
            episode_return = 0
            obs, _ = env.reset()
            episode_over = False
            while not episode_over:
                action = agent.take_action(obs)
                next_obs, r, terminated, truncated, _ = env.step(action)
                next_action = agent.take_action(next_obs)
                episode_return += r
                agent.update(obs,action,r,next_obs,next_action)
                obs = next_obs
                episode_over = terminated or truncated
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode':'%d' % (Config.num_episodes / 10 * i + i_episode + 1),'return': '%.3f' % np.mean(return_list)})
            pbar.update(1)

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