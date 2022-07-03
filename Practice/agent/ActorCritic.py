import torch as T
import numpy as np
import torch.nn.functional as F
from networks.ac_net import Actor
from networks.ac_net import Critic
from networks.ac_net import ActorCriticNetwork

# Without Baseline
class SimpleAgent():
    def __init__(self, n_inputs, n_actions, lr = 5e-6, gamma = 0.99) -> None:
        self.gamma = gamma
        self.log_pi = None
        self.actor_critic = ActorCriticNetwork(n_inputs=n_inputs, n_actions=n_actions, lr=lr)

    def choose_action(self, state):
        state = T.tensor(np.array([state]))
        pi, _ = self.actor_critic(state)
        pi_probability = F.softmax(pi, dim=1)
        action_distribution = T.distributions.Categorical(pi_probability)
        action = action_distribution.sample()
        self.log_pi = action_distribution.log_prob(action)

        return action.item()

    def learn(self, state, reward, next_state, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor(np.array([state]), dtype=T.float)
        next_state = T.tensor(np.array([next_state]), dtype=T.float)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, q = self.actor_critic(state)
        _, next_q = self.actor_critic(next_state)

        td_loss = reward + self.gamma * next_q * (1-int(done)) - q

        action_loss = - self.log_pi * td_loss
        critic_loss = td_loss ** 2

        
        (action_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()

# With Baseline Deduction
class Agent():
    def __init__(self, n_inputs, n_actions, actor_lr = 0.001, critic_lr = 0.0001, gamma = 0.99) -> None:
        self.log_values = []
        self.q_values = []
        self.states = []
        self.next_states = []
        self.rewards = []
        self.dones = []

        self.gamma = gamma
        self.actor_net = Actor(n_inputs, n_actions, actor_lr)
        self.critic_net = Critic(n_inputs, critic_lr)

    def choose_action(self, state):
        state = T.tensor([state])
        pi, value = self.actor_net(state), self.critic_net(state)
        action = pi.sample()
        log_value = pi.log_prob(action)
        self.log_values.append(log_value)
        self.q_values.append(value)

        return action.item()

    def store(self, state, next_state, reward, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def reset(self):
        self.log_values = []
        self.q_values = []
        self.states = []
        self.next_states = []
        self.rewards = []
        self.dones = []
    
    def state_values(self, next_value, gamma = 0.99):
        V = []
        V_t = next_value
        for t in reversed(range(len(self.rewards))):
            V_t = self.rewards[t] + gamma * V_t * self.dones[t]
            V.insert(0, V_t)
        return V

    def learn(self):
        next_sate = T.tensor(self.next_states[-1])
        next_value = self.critic_net(next_sate)

        V = self.state_values(next_value=next_value)

        V = T.cat(V).detach()
        q_values = T.cat(self.q_values)
        log_values = T.cat(self.log_values)

        advantages = V - q_values
        actor_loss = - (log_values * advantages.detach()).mean()

        self.actor_net.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_net.optimizer.step()

        self.critic_net.optimizer.zero_grad()
        critic_loss = advantages.pow(2).mean()
        critic_loss.backward()
        self.critic_net.optimizer.step()
        '''
        for i in range(len(log_values) - 1):
            self.actor_net.optimizer.zero_grad()
            actor_loss = - advantages[i] * log_values[i]
            actor_loss.backward()
            self.actor_net.step()

            self.critic_net.optimizer.zero_grad()
            critic_loss = self.rewards[i] + self.gamma * self.q_values[i+1] - self.q_values[i]
            critic_loss.backward()
            self.critic_net.step()
        '''
        


