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
from mappo_agent import Agent, ReplayBuffer
from common.rl_utils import evaluate_policy_mappo

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
agent_num = env_info["n_agents"]
Config.agent_num = agent_num
Config.state_dim = env_info["state_shape"]
Config.obs_dim_n = [env_info["obs_shape"]] * agent_num
Config.action_dim_n = [env_info["n_actions"]] * agent_num
Config.episode_limit = env_info["episode_limit"]

'''初始化智能体与回放池'''
agent_n = []
for agent_id in range(agent_num):
    agent_n.append(Agent(agent_id))
replay_buffer = ReplayBuffer()

'''初始化tensorboard'''
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
logdir = f"runs/{algo_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)

total_steps = 0
evaluate_num = -1
while total_steps < Config.Max_train_steps:
    ''' Eval & Record '''
    # if total_steps // Config.eval_interval > evaluate_num:
    #     win_rate, evaluate_reward = evaluate_policy_mappo(eval_env, agent_num, agent_n, Config.episode_limit, turns=32)
    #     writer.add_scalar(f'win_rate', win_rate, global_step=total_steps)
    #     writer.add_scalar(f'ep_r', evaluate_reward, global_step=total_steps)
    #     evaluate_num += 1

    done = False
    env.reset()
    for episode_step in range(Config.episode_limit):
        if done: break
        obs_n = env.get_obs()  # obs_n.shape=(N,obs_dim)
        state = env.get_state()  # s.shape=(state_dim,)
        avail_a_n = env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)

        a_list = [agent_n[i].take_action(obs_n[i],avail_a_n[i]) for i in range(agent_num)]
        a_n, logprob_a_n = map(np.array, zip(*a_list))
        ''' Get the state values (V(s)) of N agents '''
        v_n = [agent_n[i].get_value(state, obs_n[i]) for i in range(agent_num)]
        r, done, info = env.step(a_n)
        if done and episode_step + 1 != Config.episode_limit:
            dw = True
        else:
            dw = False
        ''' Store the transition '''
        replay_buffer.store_transition(episode_step, obs_n, state, v_n, avail_a_n, a_n, logprob_a_n, r, dw)

    # An episode is over, store obs_n, s and avail_a_n in the last step
    obs_n = env.get_obs()
    state = env.get_state()
    v_n = [agent_n[i].get_value(state, obs_n[i]) for i in range(agent_num)]
    replay_buffer.store_last_value(episode_step + 1, v_n)
    
env.close()