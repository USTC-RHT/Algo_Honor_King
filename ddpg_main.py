from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
value_base_dir = os.path.join(root, "model_free", "value_base")
policy_base_dir = os.path.join(root, "model_free", "policy_base")
from common.rl_utils import moving_average

algo_name = 'ddpg'
root1 = os.path.abspath(os.path.join(policy_base_dir, "ddpg"))
sys.path.append(root1)
from ddpg_config import Config
from ddpg_agent import Agent, ReplayBuffer

env_name = "CartPole-v0"
env = gym.make(env_name)
agent = Agent(env)
replaybuffer = ReplayBuffer(Config.buffer_size)
max_action = float(env.action_space.high[0])

return_list = []
total_steps = 0
for i in range(10):
    with tqdm(total=int(Config.num_episodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(Config.num_episodes/10)):
            episode_return = 0
            state, _ = env.reset()
            done = False
            Episode = []
            while not done:
                if total_steps < Config.minimal_size: 
                    # steps for random policy to explore 随机探索阶段
                    action = env.action_space.sample()
                else: 
                    action = agent.take_action(state)
                next_state, reward, dw, truncated, _ = env.step(action)
                done = dw or truncated
                replaybuffer.add(state, action, reward, next_state, dw)
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replaybuffer.size() > Config.minimal_size:
                    transitions = replaybuffer.sample(Config.batch_size)
                    agent.update(transitions)
                state = next_state
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (Config.num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
# # 设置纵坐标范围为 -250 到 0
# plt.ylim(-250, 0)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'{algo_name} on {env_name}')
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'{algo_name} on {env_name}')
plt.show()

env = gym.make(env_name,render_mode = 'human')
state, _ = env.reset()
action = agent.best_action(state)    
done = False
while not done:
    state, r, terminated, truncated, _ = env.step(action)
    next_action = agent.best_action(state)
    action = next_action
    done = terminated or truncated
env.close()