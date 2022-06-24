import gym
from agent.dqn import DQN

env = gym.make("LunarLander-v2")
env.action_space.seed(42)

dqn = DQN(env=env)
state, info = env.reset(seed=42, return_info=True)
total_rewards = 0

for _ in range(100000):
    action = dqn.choose_action(state)
    # print(action)
    state_, reward, done, info = env.step(action)
    total_rewards += reward

    env.render()
    # print(env.action_space.sample())
    dqn.store(state, action, state_, reward, done)

    if dqn.mem_count > dqn.max_mem:
        dqn.learn()

    state = state_
    if done:
        observation, info = env.reset(return_info=True)
        print(total_rewards)
        total_rewards = 0

env.close()