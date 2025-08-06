import os
import sys
import torch

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
from common.rl_utils import evaluate_policy_mappo, Normalization

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
agent_n = Agent()
replay_buffer = ReplayBuffer()

'''初始化tensorboard'''
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
logdir = f"runs/{algo_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)

total_steps = 0
evaluate_num = -1
reward_norm = Normalization(shape = 1)

while total_steps < Config.Max_train_steps:
    ''' Eval & Record '''
    if total_steps // Config.eval_interval > evaluate_num:
        win_rate, evaluate_reward = evaluate_policy_mappo(eval_env, Config.use_rnn, agent_n, Config.episode_limit, turns=32)
        writer.add_scalar(f'win_rate', win_rate, global_step=total_steps)
        writer.add_scalar(f'ep_r', evaluate_reward, global_step=total_steps)
        evaluate_num += 1

    done = False
    env.reset()
    if Config.use_rnn:
        agent_n.actor.rnn_hidden = None
        agent_n.critic.rnn_hidden = None
    for episode_step in range(Config.episode_limit):
        if done: break
        obs_n = env.get_obs()  # obs_n.shape=(N,obs_dim)
        state = env.get_state()  # s.shape=(state_dim,)
        avail_a_n = env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)

        a_n, logprob_a_n = agent_n.take_action(obs_n,avail_a_n)
        ''' Get the state values (V(s)) of N agents '''
        v_n = agent_n.get_value(state, obs_n)
        r, done, info = env.step(a_n)
        if Config.use_reward_norm:
            r = reward_norm.normalize(r)
        if done and episode_step + 1 != Config.episode_limit:
            dw = True
        else:
            dw = False
        total_steps += 1
        ''' Store the transition '''
        replay_buffer.store_transition(episode_step, obs_n, state, v_n, avail_a_n, a_n, logprob_a_n, r, dw)


    obs_n = env.get_obs()
    state = env.get_state()
    v_n = agent_n.get_value(state, obs_n)
    replay_buffer.store_last_value(episode_step + 1, v_n)

    if replay_buffer.episode_num == Config.batch_size:
        batch = replay_buffer.get_training_data()
        actor_loss, critic_loss = agent_n.update(batch,total_steps)
        print(f'actor_loss:{actor_loss}')
        print(f'critic_loss:{critic_loss}')
        replay_buffer.reset_buffer()

    
env.close()