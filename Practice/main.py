import gym
import numpy as np
from agent import ActorCritic
from agent import REINFORCE
from agent import DDPG

env = gym.make("LunarLander-v2")
env_continuous = gym.make("LunarLanderContinuous-v2")

n_games = 2000

env.action_space.seed(42)

# agent = Agent(n_actions=env.action_space.n, n_inputs=[env.observation_space.shape[0]], gamma=0.99)

## ActorCritic
ac_agent = ActorCritic.SimpleAgent(env=env, n_inputs=[env.observation_space.shape[0]], n_actions=env.action_space.n)
# ac_agent.run(n_games=n_games)

## REINFORCE
agent = REINFORCE.Agent(env=env, n_actions=env.action_space.n, n_inputs=[env.observation_space.shape[0]], lr= 0.0005, gamma=0.99)
# agent.run(n_games=n_games)

## DDPG
ddpg_agent = DDPG.Agent(n_inputs=env.observation_space.shape, n_actions=env.action_space.shape[0])
# ddpg_agent.run(n_games=n_games)