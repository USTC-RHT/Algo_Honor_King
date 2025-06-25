from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import sys
cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
value_base_dir = os.path.join(root, "model_free", "value_base")

prioritized_replay = False
# algo_name = 'dqn'
# algo_name = 'double_dqn'
# algo_name = 'dueling_dqn'
algo_name = 'prioritized_dqn'

if algo_name == 'dqn':
    root1 = os.path.abspath(os.path.join(value_base_dir, "dqn"))
    sys.path.append(root1)
    from dqn_config import Config
    from dqn_agent import Agent,ReplayBuffer
elif algo_name == 'double_dqn':
    root1 = os.path.abspath(os.path.join(value_base_dir, "double_dqn"))
    sys.path.append(root1)
    from double_dqn_config import Config
    from double_dqn_agent import Agent,ReplayBuffer
elif algo_name == 'dueling_dqn':
    root1 = os.path.abspath(os.path.join(value_base_dir, "dueling_dqn"))
    sys.path.append(root1)
    from dueling_dqn_config import Config
    from dueling_dqn_agent import Agent,ReplayBuffer
elif algo_name == 'prioritized_dqn':
    root1 = os.path.abspath(os.path.join(value_base_dir, "prioritized_dqn"))
    sys.path.append(root1)
    from prioritized_dqn_config import Config
    from prioritized_dqn_agent import Agent,PrioritizedReplayBuffer
    prioritized_replay = True


# env_name = "CliffWalking-v0"    # "FrozenLake-v1"
# env_name = 'Pendulum-v1'
env_name = "CartPole-v0"
env = gym.make(env_name)

agent = Agent(env)
if algo_name == 'prioritized_dqn':
    replaybuffer = PrioritizedReplayBuffer(Config.buffer_size, Config.alpha)
else:  
    replaybuffer = ReplayBuffer(Config.buffer_size)

return_list = []
for i in range(10):
    with tqdm(total=int(Config.num_episodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(Config.num_episodes/10)):
            episode_return = 0
            obs, _ = env.reset()
            done = False
            Episode = []
            while not done:
                action = agent.take_action(obs)
                next_obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replaybuffer.add(obs, action, r, next_obs, done)
                episode_return += r 
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replaybuffer.size() > Config.minimal_size:
                    if prioritized_replay:
                        transitions, weights, batch_idxes = replaybuffer.sample(Config.batch_size, beta=Config.beta)
                        td_errors = agent.update(transitions, weights)
                    else:                    
                        transitions = replaybuffer.sample(Config.batch_size)
                        agent.update(transitions)
                    if prioritized_replay:
                        new_priorities = np.abs(td_errors) + Config.prioritized_replay_eps
                        replaybuffer.update_priorities(batch_idxes, new_priorities)
                obs = next_obs
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (Config.num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
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