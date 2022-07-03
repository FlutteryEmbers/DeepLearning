import gym
import numpy as np
from agent.DDPG import Agent

env = gym.make("LunarLanderContinuous-v2")
n_games = 1000

env.action_space.seed(42)

# agent = Agent(n_actions=env.action_space.n, n_inputs=[env.observation_space.shape[0]], gamma=0.99)
simple_agent = Agent(n_inputs=env.observation_space.shape, n_actions=env.action_space.shape[0])
pr
scores = []
for i in range(n_games):
    score = 0
    done = False
    state = env.reset()
    simple_agent.noise.reset()

    while not done:
        action = simple_agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        score += reward
        env.render()

        simple_agent.store_transition(state=state, action=action, next_state=state_, reward=reward, done=done)
        simple_agent.learn()
        state = state_

    # agent.learn()
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print('episode', i, 'score %.1f' % score, 'average score %.2f' % avg_score)

env.close()