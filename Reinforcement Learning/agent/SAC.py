import torch as T
import torch.nn.functional as F
from networks.sac_net import Actor, Critic, Value
from utils.buffer import ReplayBuffer
import numpy as np

class Agent():
    def __init__(self, n_inputs, n_actions, max_action, tau = 0.005, gamma = 0.99, actor_lr = 3e-4, critic_lr = 3e-4, value_lr = 3e-4, buffer_size = 10e6, batch_size = 256) -> None:
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.buffer = ReplayBuffer(max_size=buffer_size, n_actions=n_actions, input_shape=n_inputs)
        self.actor = Actor(n_inputs=n_inputs, n_actions=n_actions, max_action=max_action, lr=actor_lr)
        self.critic_1 = Critic(n_inputs=n_inputs, n_actions=n_actions, lr=critic_lr)
        self.critic_2 = Critic(n_inputs=n_inputs, n_actions=n_actions, lr=critic_lr)
        self.value = Value(n_inputs=n_inputs, lr=value_lr)
        self.target_value = Value(n_inputs=n_inputs, lr=value_lr)
    
    def store(self, state, action, next_state, reward, done):
        self.buffer.store(state, action, next_state, reward, done)

    def choose_action(self, state):
        state = T.tensor(state)
        action, log_prob = self.actor.sample_normal(state, reparameterize=False)

        return action.item()

    def learn(self):
        if self.buffer.mem_count < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.buffer.sample(batch_size=self.batch_size)
        states = T.tensor(np.array(states)).to(self.actor.device)
        actions = T.tensor(np.array(actions)).to(self.actor.device)
        rewards = T.tensor(np.array(rewards)).to(self.actor.device)
        states_ = T.tensor(np.array(states_)).to(self.actor.device)
        dones = T.tensor(np.array(dones)).to(self.actor.device)

        value = self.value(states).view(-1)
        values_ = self.target_value(states_).view(-1)
        values_[dones] = 0.0

        actions, log_probs = self.actor.sample_normal(states, reparameterize=False)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1(states, actions)
        q2_new_policy = self.critic_2(states, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optim.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optim.step()

        q_hat = rewards + self.gamma * values_
        q1_old_policy = self.critic_1(states, actions).view(-1)
        q2_old_policy = self.critic_2(states, actions).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        self.critic_1.optim.zero_grad()
        self.critic_2.optim.zero_grad()
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_1.optim.step()
        self.critic_2.optim.step()

        actions, log_probs = self.actor.sample_normal(states, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states, actions)
        q2_new_policy = self.critic_2(states, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optim.step()

        self.update_network_parameter()
    
    def update_network_parameter(self, tau = None):
        tau = 0 if tau == None else self.tau
        

