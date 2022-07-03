from networks.ddpg_net import Actor
from networks.ddpg_net import Critic
from utils.buffer import ReplayBuffer
from utils.OUAnoise import OUActionNoise

class Agent():
    def __init__(self, state_dims, n_actions, tau = 0.001, gamma = 0.99, buffer_size = 1e6, batch_size = 64) -> None:
        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.actor = Actor(n_inputs=state_dims, n_actions=n_actions)
        self.actor_target = Actor(n_inputs=state_dims, n_actions=n_actions)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(n_inputs=state_dims, n_actions=n_actions)
        self.critic_target = Critic(n_inputs=state_dims, n_actions=n_actions)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = ReplayBuffer(max_size = buffer_size, input_shape = state_dims, n_actions = n_actions)

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state=state, action=action, state_=next_state, rewards=reward, done=done)

    def choose_action(self, state):
        pass

