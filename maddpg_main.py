import os
import sys
import torch
import copy
import numpy as np

'''MPE 环境指的是 Multi-Agent Particle Environment,是多智能体强化学习中
一个经典的测试平台,由 OpenAI 提出,用于研究多智能体协作与竞争问题.'''
from make_env import make_env

cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
policy_base_dir = os.path.join(root, "model_free", "policy_base")

algo_name = 'maddpg'
root1 = os.path.abspath(os.path.join(policy_base_dir, "maddpg"))
sys.path.append(root1)
from maddpg_config import Config
from maddpg_agent import Agent, ReplayBuffer
from common.rl_utils import evaluate_policy_maddpg

env_name = 'simple_spread'
# env_name = 'simple_adversary'
env = make_env(env_name, discrete=False)
eval_env = make_env(env_name, discrete=False)
'''设置随机数种子,提升训练的可复现性'''
seed = 0
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''初始化智能体与回放池'''
agent_n = []
agent_num = env.n  # The number of agents
Config.agent_num = agent_num
Config.obs_dim_n = [env.observation_space[i].shape[0] for i in range(agent_num)]  # obs dimensions of N agents
Config.action_dim_n = [env.action_space[i].shape[0] for i in range(agent_num)]  # actions dimensions of N agents

for agent_id in range(agent_num):
    agent_n.append(Agent(agent_id))
replaybuffer = ReplayBuffer(Config)

'''初始化tensorboard'''
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
logdir = f"runs/{algo_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)

noise_decay = (Config.noise_init - Config.noise_min) / Config.noise_decay_steps

total_steps = 0
score_list = evaluate_policy_maddpg(eval_env, agent_num, agent_n, 3, Config.episode_limit)
print(f'total_steps{total_steps}:{score_list[0]}')
for i in range(agent_num):
    writer.add_scalar(f'{i}/ep_r', score_list[i], global_step=total_steps)

while total_steps < Config.Max_train_steps:
    episode_return = 0
    '''obs_n是list类型 以agent_id为索引 每个元素均为numpy数组'''
    obs_n = env.reset()
    done = False
    for _ in range(Config.episode_limit):
        if done: break

        action_n = [agent_n[i].take_action(obs_n[i]).astype(np.float32) for i in range(agent_num)]
        next_obs_n, reward_n, dw_n, _ = env.step(copy.deepcopy(action_n))
        done = any(dw_n)
        replaybuffer.store_transition(obs_n, action_n, reward_n, next_obs_n, dw_n)

        # Decay noise
        if Config.use_noise_decay and Config.noise > Config.noise_min:
            Config.noise -= noise_decay
            for i in range(agent_num):
                agent_n[i].noise = Config.noise

        # 当buffer数据的数量超过一定值后进行训练
        if replaybuffer.current_size > Config.minimal_size \
        and total_steps % Config.update_every == 0:
            transitions = replaybuffer.sample()
            for i in range(agent_num):
                actor_loss, critic_loss = agent_n[i].update(transitions,agent_n)
                writer.add_scalar(f'{i}/actor_loss', actor_loss, global_step=total_steps)
                writer.add_scalar(f'{i}/critic_loss', critic_loss, global_step=total_steps)

        obs_n = next_obs_n
        total_steps += 1
        '''Eval & Record 
        ep_r: episode reward'''
        if total_steps % Config.eval_interval == 0:
            score_list = evaluate_policy_maddpg(eval_env, agent_num, agent_n, 3, Config.episode_limit)
            print(f'total_steps{total_steps}:{score_list[0]}')
            for i in range(agent_num):
                writer.add_scalar(f'{i}/ep_r', score_list[i], global_step=total_steps)
env.close()