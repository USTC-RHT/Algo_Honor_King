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

# algo_name = 'monte_carlo'
# algo_name = 'sarsa'
# algo_name = 'q_learning'
# algo_name = 'reinforce'
algo_name = 'a2c'

if algo_name == 'monte_carlo':
    root1 = os.path.abspath(os.path.join(value_base_dir, "monte_carlo"))
    sys.path.append(root1)
    from monte_carlo_config import Config
    from monte_carlo_agent import Agent
if algo_name == 'sarsa':
    root1 = os.path.abspath(os.path.join(value_base_dir, "sarsa"))
    sys.path.append(root1)
    from sarsa_config import Config
    from sarsa_agent import Agent
if algo_name == 'q_learning':
    root1 = os.path.abspath(os.path.join(value_base_dir, "q_learning"))
    sys.path.append(root1)
    from q_learning_config import Config
    from q_learning_agent import Agent
if algo_name == 'reinforce':
    root1 = os.path.abspath(os.path.join(policy_base_dir, "reinforce"))
    sys.path.append(root1)
    from reinforce_config import Config
    from reinforce_agent import Agent
if algo_name == 'a2c':
    root1 = os.path.abspath(os.path.join(policy_base_dir, "a2c"))
    sys.path.append(root1)
    from a2c_config import Config
    from a2c_agent import Agent


# env_name = "CliffWalking-v0"    # "FrozenLake-v1"
env_name = "CartPole-v1"
# env_name = "Acrobot-v1"
env = gym.make(env_name)
if algo_name in ['reinforce','dqn','a2c']:
    agent = Agent(env)
else:
    agent = Agent(env_name)


return_list = []
total_steps = 0
for i in range(10):
    with tqdm(total=int(Config.num_episodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(Config.num_episodes/10)):
            episode_return = 0
            obs, _ = env.reset()
            done = False
            Episode = []
            while not done:
                action = agent.take_action(obs)
                next_obs, r, dw, truncated, _ = env.step(action)
                done = dw or truncated
                episode_return += r 
                total_steps += 1
                if algo_name in ['monte_carlo','reinforce']:
                    Episode.append((obs,action,r))      
                elif algo_name == 'sarsa':
                    next_action = agent.take_action(next_obs)
                    agent.update(obs,action,r,next_obs,next_action,done)                
                elif algo_name == 'q_learning':
                    agent.update(obs,action,r,next_obs,done)
                elif algo_name == 'a2c':
                    Episode.append((obs,action,r,next_obs,dw))
                obs = next_obs
            if algo_name in ['monte_carlo','reinforce','a2c']:
                agent.update(Episode)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (Config.num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

# print(total_steps)
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
obs, _ = env.reset()
action = agent.best_action(obs)    
episode_over = False
while not episode_over:
    obs, r, terminated, truncated, _ = env.step(action)
    next_action = agent.best_action(obs)
    action = next_action
    episode_over = terminated or truncated
env.close()