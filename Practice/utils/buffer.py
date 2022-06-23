import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, *input_shape, n_actions) -> None:
        self.memory_size = max_size
        self.mem_count = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.next_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros((self.memory_size, n_actions))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)
        

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def store(self, state, action, state_, rewards, done):
        idx = self.mem_count % self.memory_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.next_state_memory[idx] = state_
        self.reward_memory[idx] = rewards
        self.terminal_memory[idx] = done
        
        self.mem_count += 1