import gym
import numpy as np
from agent.ActorCritic import Agent
from agent.ActorCritic import SimpleAgent

env = gym.make("LunarLander-v2")
n_games = 2000

env.action_space.seed(42)

# agent = Agent(n_actions=env.action_space.n, n_inputs=[env.observation_space.shape[0]], gamma=0.99)
simple_agent = SimpleAgent(n_inputs=[env.observation_space.shape[0]], n_actions=env.action_space.n)

scores = []
for i in range(n_games):
    score = 0
    state = env.reset()
    done = False
    # agent.reset()
    while not done:
        # action = agent.choose_action(state)
        action = simple_agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        score += reward
        env.render()
        # agent.store(state=state, next_state=state_, reward=reward, done=done)
        simple_agent.learn(state=state, reward=reward, next_state=state_, done=done)
        state = state_

    # agent.learn()
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print('episode', i, 'score %.1f' % score, 'average score %.2f' % avg_score)

env.close()