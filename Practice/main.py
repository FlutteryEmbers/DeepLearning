import gym
import numpy as np
from agent.REINFORCE import Agent

env = gym.make("LunarLander-v2")
n_games = 3000

env.action_space.seed(42)

agent = Agent(n_actions=env.action_space.n, n_inputs=[env.observation_space.shape[0]], lr= 0.0005, gamma=0.99)

scores = []
for i in range(n_games):
    score = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        score += reward
        env.render()
        agent.store(reward=reward)
        state = state_

    agent.learn()
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print('episode', i, 'score %.1f' % score, 'average score %.2f' % avg_score)

env.close()