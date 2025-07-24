import os
import sys
import torch
import copy
import numpy as np

'''MPE 环境指的是 Multi-Agent Particle Environment,是多智能体强化学习中
一个经典的测试平台,由 OpenAI 提出,用于研究多智能体协作与竞争问题.'''
from mpe2 import simple_adversary_v3, simple_speaker_listener_v4

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

env_name = 'simple_adversary'
env_name = 'simple_speaker_listener'
env = simple_speaker_listener_v4.parallel_env(continuous_actions = True)
eval_env = simple_speaker_listener_v4.parallel_env(continuous_actions = True)
'''设置随机数种子,提升训练的可复现性'''
seed = 0
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''初始化智能体与回放池'''
agent_n = []
agent_names = env.possible_agents
Config.agent_names = agent_names
Config.min_action = env.action_space(agent_names[0]).low[0].item()
Config.max_action = env.action_space(agent_names[0]).high[0].item()
for agent_name in agent_names:
    agent_n.append(Agent(env,agent_name))
replaybuffer = ReplayBuffer(Config.buffer_size)

'''初始化tensorboard'''
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
logdir = f"runs/{algo_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)

noise_decay = (Config.noise_init - Config.noise_min) / Config.noise_decay_steps

total_steps = 0
while total_steps < Config.Max_train_steps:
    episode_return = 0
    obs_n, _ = env.reset()
    env_seed += 1
    done = False
    for _ in range(Config.episode_limit):
        while not done:
            # if total_steps < Config.minimal_size: 
            #     # steps for random policy to explore 随机探索阶段
            #     action_n = {agent_name: env.action_space(agent_name).sample() \
            #                for agent_name in agent_names}

            action_n = {agent_name: agent_n[i].take_action(obs_n[agent_name]).astype(np.float32) \
                        for i,agent_name in enumerate(agent_names)}

            next_obs_n, reward_n, dw_n, truncated_n, _ = env.step(copy.deepcopy(action_n))
            done = any(dw_n.values()) or any(truncated_n.values())
            replaybuffer.add(obs_n, action_n, reward_n, next_obs_n, dw_n)
            # 当buffer数据的数量超过一定值后进行训练
            if replaybuffer.size() >= Config.minimal_size \
            and total_steps % Config.update_every == 0:
                transitions = replaybuffer.sample(Config.batch_size)
                for i,agent_name in enumerate(agent_names):
                    actor_loss, critic_loss = agent_n[i].update(transitions,agent_n)
                    writer.add_scalar(f'{agent_name}/actor_loss', actor_loss, global_step=total_steps)
                    writer.add_scalar(f'{agent_name}/critic_loss', critic_loss, global_step=total_steps)
            obs_n = next_obs_n
            # Decay noise
            if Config.use_noise_decay and Config.noise > Config.noise_min:
                Config.noise -= noise_decay
                for i in range(len(agent_names)):
                    agent_n[i].noise = Config.noise
            total_steps += 1
            '''Eval & Record 
            ep_r: episode reward'''
            if total_steps % Config.eval_interval == 0:
                score_list = evaluate_policy_maddpg(eval_env, agent_names, agent_n, 3, Config.episode_limit)
                for i in range(len(agent_names)):
                    writer.add_scalar(f'{agent_names[i]}/ep_r', score_list[i], global_step=total_steps)
env.close()