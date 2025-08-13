import gymnasium as gym
import os
import sys
import torch
cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
policy_base_dir = os.path.join(root, "model_free", "policy_base")
from common.rl_utils import evaluate_policy, reward_shaping
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='LunarLanderContinuous-v3', help='环境名称')
parser.add_argument('--seed', type=int, default=0, help='随机种子')
args = parser.parse_args()

print(f"环境: {args.env_name}, 随机种子: {args.seed}")

algo_name = 'td3'
root = os.path.abspath(os.path.join(policy_base_dir, "td3"))
sys.path.append(root)
from td3_config import Config
from td3_agent import Agent, ReplayBuffer

# env_name = 'Pendulum-v1'
# env_name = 'LunarLanderContinuous-v3'
# env_name = 'BipedalWalkerHardcore-v3'
env_name = args.env_name
env = gym.make(env_name)
eval_env = gym.make(env_name)
'''设置随机数种子,提升训练的可复现性'''
seed = args.seed
env_seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
max_ep_steps = env.spec.max_episode_steps
max_action = float(env.action_space.high[0])
'''初始化智能体与回放池'''
agent = Agent(env)
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
        if total_steps < 10 * max_ep_steps: 
            # steps for random policy to explore 随机探索阶段
            action = env.action_space.sample()
        else: 
            action = agent.predict(state)
        next_state, reward, dw, truncated, _ = env.step(action)
        reward = reward_shaping(reward, env_name)
        done = dw or truncated
        replaybuffer.add(state, action, reward, next_state, dw)
        episode_return += reward
        if total_steps >= 2 * max_ep_steps and total_steps % Config.update_every == 0:
            actor_loss = None
            critic_loss = None
            for _ in range(Config.update_every):
                transitions = replaybuffer.sample(Config.batch_size)
                a_loss, c_loss = agent.learn(transitions)
                if actor_loss is None:
                    actor_loss = a_loss
                    critic_loss = c_loss
            writer.add_scalar('actor_loss', actor_loss, global_step=total_steps)
            writer.add_scalar('critic_loss', critic_loss, global_step=total_steps)
        state = next_state
        total_steps += 1
        '''Eval & Record 
        ep_r: episode reward'''
        if total_steps % Config.eval_interval == 0:
            '''探索噪声衰减'''
            agent.explore_noise *= Config.explore_noise_decay
            score = evaluate_policy(eval_env, agent, turns=3)
            writer.add_scalar('ep_r', score, global_step=total_steps)

'''Save model'''
torch.save({'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),}, f'models/{algo_name}_{env_name}_model.pth')

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