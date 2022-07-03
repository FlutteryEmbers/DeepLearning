import torch as T
import torch.nn.functional as F
import numpy as np
from networks.ddpg_net import Actor, Critic
from utils.buffer import ReplayBuffer
from utils.OUAnoise import OUActionNoise

class Agent():
    def __init__(self, n_inputs, n_actions, tau = 0.001, gamma = 0.99, buffer_size = int(1e6), batch_size = 64) -> None:
        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.actor = Actor(n_inputs=n_inputs, n_actions=n_actions)
        self.actor_target = Actor(n_inputs=n_inputs, n_actions=n_actions)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(n_inputs=n_inputs, n_actions=n_actions)
        self.critic_target = Critic(n_inputs=n_inputs, n_actions=n_actions)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = ReplayBuffer(max_size = buffer_size, input_shape = n_inputs, n_actions = n_actions)
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state=state, action=action, state_=next_state, rewards=reward, done=done)

    def choose_action(self, state):
        self.actor.eval()
        state = T.tensor(np.array([state]))
        mu = self.actor(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype = T.float).to(self.actor.device)

        self.actor.train()
        
        return mu_prime.cpu().detach().numpy()[0]

    def learn(self):
        if self.replay_buffer.mem_count < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.replay_buffer.sample(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        

        target_actions = self.actor_target(states_)
        next_q = self.critic_target(states_, target_actions)
        q = self.critic(states, actions)

        next_q[dones] = 0.0
        next_q = next_q.view(-1)

        target = rewards + self.gamma * next_q
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, q)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        actor_target_params = self.actor_target.named_parameters()
        critic_target_params = self.critic_target.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        actor_target_state_dict = dict(actor_target_params)
        critic_target_state_dict = dict(critic_target_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*critic_target_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*actor_target_state_dict[name].clone()

        self.actor_target.load_state_dict(actor_state_dict)
        self.critic_target.load_state_dict(critic_state_dict)
        


    

