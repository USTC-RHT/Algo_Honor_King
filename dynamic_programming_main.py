from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import sys
cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)

model_base_dir = os.path.join(root, "model_based")

algo_name = 'dynamic_programming'
if algo_name == 'dynamic_programming':
    root1 = os.path.abspath(os.path.join(model_base_dir, "given_model", "dynamic_programming"))
    sys.path.append(root1)
    from dynamic_programming_config import Config
    from dynamic_programming_agent import Agent


env_name = "CliffWalking-v0"    # "FrozenLake-v1"
env = gym.make(env_name)
agent = Agent(env_name)

agent.update()

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