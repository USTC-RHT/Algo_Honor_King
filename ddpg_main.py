import gymnasium as gym
import os
import sys
import torch
cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
value_base_dir = os.path.join(root, "model_free", "value_base")
policy_base_dir = os.path.join(root, "model_free", "policy_base")
from common.rl_utils import evaluate_policy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


algo_name = 'ddpg'
root1 = os.path.abspath(os.path.join(policy_base_dir, "ddpg"))
sys.path.append(root1)
from ddpg_config import Config
from ddpg_agent import Agent, ReplayBuffer

# env_name = 'Pendulum-v1'
env_name = 'LunarLanderContinuous-v3'
env = gym.make(env_name)
eval_env = gym.make(env_name)
'''设置随机数种子,提升训练的可复现性'''
seed = 0
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
max_action = float(env.action_space.high[0])
'''初始化智能体与回放池'''
agent = Agent(env,max_action)
replaybuffer = ReplayBuffer(Config.buffer_size)

logdir = f"runs/{algo_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)

total_steps = 0
while total_steps < Config.Max_train_steps:
    episode_return = 0
    state, _ = env.reset(seed=env_seed)
    env_seed += 1
    done = False
    while not done:
        if total_steps < Config.minimal_size: 
            # steps for random policy to explore 随机探索阶段
            action = env.action_space.sample()
        else: 
            action = agent.predict(state)
        next_state, reward, dw, truncated, _ = env.step(action)
        done = dw or truncated
        replaybuffer.add(state, action, reward, next_state, dw)
        episode_return += reward
        # 当buffer数据的数量超过一定值后进行训练
        if replaybuffer.size() > Config.minimal_size:
            transitions = replaybuffer.sample(Config.batch_size)
            actor_loss, critic_loss = agent.learn(transitions)
            writer.add_scalar('actor_loss', actor_loss, global_step=total_steps)
            writer.add_scalar('critic_loss', critic_loss, global_step=total_steps)
        state = next_state
        total_steps += 1
        '''Eval & Record 
        ep_r: episode reward'''
        if total_steps % Config.eval_interval == 0:
            score = evaluate_policy(eval_env, agent, turns=10)
            writer.add_scalar('ep_r', score, global_step=total_steps)

env = gym.make(env_name,render_mode = 'human')
state, _ = env.reset()
action = agent.exploit(state)    
done = False
while not done:
    state, r, terminated, truncated, _ = env.step(action)
    next_action = agent.exploit(state)
    action = next_action
    done = terminated or truncated
env.close()