import os
import sys
import torch
import copy
import numpy as np

'''StarCraft2Env 是一个为多智能体强化学习设计的仿真环境,来源于SMAC(StarCraft Multi-Agent Challenge)。
它基于暴雪出品的即时战略游戏 StarCraft II,提供多个小规模战斗场景,允许研究多智能体之间的协作、对抗和策略学习。'''
from smac.env import StarCraft2Env

cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
policy_base_dir = os.path.join(root, "model_free", "policy_base")

algo_name = 'mappo'
root1 = os.path.abspath(os.path.join(policy_base_dir, "mappo"))
sys.path.append(root1)
from mappo_config import Config
from mappo_agent import Agent
from common.rl_utils import evaluate_policy_maddpg

'''设置随机数种子,提升训练的可复现性'''
seed = 0
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''初始化多智能体环境'''
env_names = ['3m', '8m', '2s3z']
env_index = 0
env_name = env_names[env_index]
env = StarCraft2Env(map_name=env_name, seed=seed)
eval_env = StarCraft2Env(map_name=env_name, seed=seed)

'''获取环境相关信息'''
env_info = env.get_env_info()
Config.agent_num = env_info["n_agents"]
Config.state_dim = env_info["state_shape"]
Config.obs_dim_n = [env_info["obs_shape"]] * Config.agent_num
Config.action_dim_n = [env_info["n_actions"]] * Config.agent_num
Config.episode_limit = env_info["episode_limit"]

'''初始化智能体与回放池'''
agent_n = []
for agent_id in range(Config.agent_num):
    agent_n.append(Agent(agent_id))

'''初始化tensorboard'''
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
logdir = f"runs/{algo_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)

noise_decay = (Config.noise_init - Config.noise_min) / Config.noise_decay_steps

total_steps = 0
while total_steps < Config.Max_train_steps:
    episode_return = 0
    obs_n = env.reset()
    env_seed += 1
    done = False
    for _ in range(Config.episode_limit):
        if done: break

        action_n = [agent_n[i].take_action(obs_n[i]).astype(np.float32) for i in range(agent_num)]
        next_obs_n, reward_n, dw_n, _ = env.step(copy.deepcopy(action_n))
        done = any(dw_n)
        replaybuffer.store_transition(obs_n, action_n, reward_n, next_obs_n, dw_n)
        # 当buffer数据的数量超过一定值后进行训练
        if replaybuffer.current_size > Config.minimal_size \
        and total_steps % Config.update_every == 0:
            transitions = replaybuffer.sample()
            for i in range(agent_num):
                actor_loss, critic_loss = agent_n[i].update(transitions,agent_n)
                writer.add_scalar(f'{i}/actor_loss', actor_loss, global_step=total_steps)
                writer.add_scalar(f'{i}/critic_loss', critic_loss, global_step=total_steps)
        obs_n = next_obs_n
        # Decay noise
        if Config.use_noise_decay and Config.noise > Config.noise_min:
            Config.noise -= noise_decay
            for i in range(agent_num):
                agent_n[i].noise = Config.noise
        total_steps += 1
        '''Eval & Record 
        ep_r: episode reward'''
        if total_steps % Config.eval_interval == 0:
            score_list = evaluate_policy_maddpg(eval_env, agent_num, agent_n, 3, Config.episode_limit)
            for i in range(agent_num):
                writer.add_scalar(f'{i}/ep_r', score_list[i], global_step=total_steps)
env.close()