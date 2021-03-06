import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.basic import PolicyNetWork

class Agent():
    def __init__(self, env, n_inputs, n_actions, gamma=0.99, lr = 0.001) -> None:
        self.gamma = gamma
        self.lr = lr
        self.env = env

        self.reward_memory = []
        self.action_memory = []
        
        self.policy_net = PolicyNetWork(n_inputs=n_inputs, n_actions=n_actions, lr=lr)

    def store(self, reward):
        self.reward_memory.append(reward)

    def choose_action(self, state):
        state = T.tensor([state]).to(self.policy_net.device)
        action_probability = F.softmax(self.policy_net(state))
        action_distribution = T.distributions.Categorical(action_probability)
        action = action_distribution.sample()
        log_action = action_distribution.log_prob(action)
        self.action_memory.append(log_action)

        return action.item()

    def learn(self):
        self.policy_net.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy_net.optimizer.step()

        self.action_memory = []
        self.reward_memory = []

    def run(self, n_games):
        scores = []
        for i in range(n_games):
            score = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                state_, reward, done, info = self.env.step(action)
                score += reward
                self.env.render()
                self.store(reward=reward)
                state = state_

            self.learn()
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print('episode', i, 'score %.1f' % score, 'average score %.2f' % avg_score)

        self.env.close()

    