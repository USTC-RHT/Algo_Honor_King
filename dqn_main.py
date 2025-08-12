import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import sys
import torch
import matplotlib.pyplot as plt
cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
value_base_dir = os.path.join(root, "model_free", "value_base")
from common.rl_utils import evaluate_policy_noisy_dqn, moving_average
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

prioritized_replay = False
# algo_name = 'dqn'
# algo_name = 'double_dqn'
algo_name = 'dueling_dqn'
# algo_name = 'prioritized_dqn'
# algo_name = 'noisy_dqn'

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
elif algo_name == 'noisy_dqn':
    root1 = os.path.abspath(os.path.join(value_base_dir, "noisy_dqn"))
    sys.path.append(root1)
    from noisy_dqn_config import Config
    from noisy_dqn_agent import Agent,ReplayBuffer
    logdir = f"runs/{algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logdir)

# env_name = "CliffWalking-v0"    # "FrozenLake-v1"
# env_name = "CartPole-v0"
# env_name = "CartPole-v1"
# env_name = 'LunarLander-v3'
env_name = 'Pendulum-v1'

env = gym.make(env_name)
eval_env = gym.make(env_name)
# 设置随机数种子,提升训练的可复现性
seed = 0
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if env_name == 'Pendulum-v1':
    Config.state_dim = env.observation_space.shape[0]
    Config.action_dim = 11  # 将连续动作分成11个离散动作
    def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
        action_lowbound = env.action_space.low[0]  # 连续动作的最小值
        action_upbound = env.action_space.high[0]  # 连续动作的最大值
        return action_lowbound + (discrete_action /
                                (action_dim - 1)) * (action_upbound - action_lowbound)
else:
    Config.state_dim = env.observation_space.shape[0]
    Config.action_dim = env.action_space.n

agent = Agent()
if algo_name == 'prioritized_dqn':
    replaybuffer = PrioritizedReplayBuffer(Config.buffer_size, Config.alpha)
else:  
    replaybuffer = ReplayBuffer(Config.buffer_size)

return_list = []
max_q_value_list = []
max_q_value = 0
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
                if env_name == 'Pendulum-v1':
                    continuous_action = dis_to_con(action,env,Config.action_dim)
                    next_obs, r, terminated, truncated, _ = env.step([continuous_action])
                else:
                    next_obs, r, terminated, truncated, _ = env.step(action)
                max_q_value = agent.max_q_value(obs) * 0.005 + max_q_value * 0.995  # 平滑处理
                max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                done = terminated or truncated
                replaybuffer.add(obs, action, r, next_obs, done)
                episode_return += r 
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replaybuffer.size() > Config.minimal_size:
                    if prioritized_replay:
                        transitions, weights, batch_idxes = replaybuffer.sample(Config.batch_size, beta=Config.beta)
                        td_errors = agent.update(transitions, weights)
                        new_priorities = np.abs(td_errors) + Config.prioritized_replay_eps
                        replaybuffer.update_priorities(batch_idxes, new_priorities)
                    else:                    
                        # train 50 times every 50 steps rather than 1 training per step.(Noisy_DQN) 
                        if algo_name == 'noisy_dqn' and total_steps % Config.update_every == 0:
                            for j in range(Config.update_every): 
                                transitions = replaybuffer.sample(Config.batch_size)
                                agent.update(transitions)
                        else:
                            transitions = replaybuffer.sample(Config.batch_size)
                            agent.update(transitions)
                if algo_name == 'noisy_dqn' and total_steps % Config.eval_interval == 0:
                    score = evaluate_policy_noisy_dqn(eval_env, agent, turns = 20)
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

episodes_list = list(range(len(return_list)))
mv_return = moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'{algo_name} on {env_name}')
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title(f'{algo_name} on {env_name}')
plt.show()