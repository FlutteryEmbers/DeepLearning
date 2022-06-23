import gym
from agent.dqn import DQN

env = gym.make("LunarLander-v2")
env.action_space.seed(42)

dqn = DQN(env=env)
state, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    action = dqn.choose_action(state)
    state_, reward, done, info = env.step(action)
    # print(observation, reward, done, info)
    env.render()
    dqn.store(state, action, state_, reward, done)
    if dqn.memory_counter > MEMORY_CAPACITY:
        dqn.learn()

    state = state_
    if done:
        observation, info = env.reset(return_info=True)

env.close()