import os
import sys
import torch

'''MPE 环境指的是 Multi-Agent Particle Environment,是多智能体强化学习中
一个经典的测试平台,由 OpenAI 提出,用于研究多智能体协作与竞争问题.'''
from mpe2 import simple_adversary_v3

cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
policy_base_dir = os.path.join(root, "model_free", "policy_base")

algo_name = 'maddpg'
root1 = os.path.abspath(os.path.join(policy_base_dir, "maddpg"))
sys.path.append(root1)
from maddpg_config import Config
from maddpg_agent import Agent, ReplayBuffer
from common.rl_utils import evaluate_policy

env_name = 'simple_adversary'
env = simple_adversary_v3.env()
eval_env = simple_adversary_v3.env()
'''设置随机数种子,提升训练的可复现性'''
seed = 0
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
max_action = float(env.action_space.high[0])

'''初始化智能体与回放池'''
agent = Agent(env)
replaybuffer = ReplayBuffer(Config.buffer_size)

'''初始化tensorboard'''
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
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
            action = agent.take_action(state)
        next_state, reward, dw, truncated, _ = env.step(action)
        done = dw or truncated
        replaybuffer.add(state, action, reward, next_state, dw)
        episode_return += reward
        # 当buffer数据的数量超过一定值后进行训练
        if replaybuffer.size() > Config.minimal_size \
        and total_steps % Config.update_every == 0:
            transitions = replaybuffer.sample(Config.batch_size)
            actor_loss, critic_loss = agent.update(transitions)
            writer.add_scalar('actor_loss', actor_loss, global_step=total_steps)
            writer.add_scalar('critic_loss', critic_loss, global_step=total_steps)
        state = next_state
        total_steps += 1
        '''Eval & Record 
        ep_r: episode reward'''
        if total_steps % Config.eval_interval == 0:
            score = evaluate_policy(eval_env, agent, turns=10)
            writer.add_scalar('ep_r', score, global_step=total_steps)
env.close()

# env = gym.make(env_name,render_mode = 'human')
# state, _ = env.reset()
# action = agent.best_action(state)    
# done = False
# while not done:
#     state, r, terminated, truncated, _ = env.step(action)
#     next_action = agent.best_action(state)
#     action = next_action
#     done = terminated or truncated
# env.close()