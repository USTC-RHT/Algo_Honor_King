from mpe2 import simple_adversary_v3

env = simple_adversary_v3.parallel_env(continuous_actions = True)
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
