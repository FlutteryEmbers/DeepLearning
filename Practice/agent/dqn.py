import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks.basic import Net
from utils.buffer import ReplayBuffer


LR = 0.01
BATCH_SIZE = 32
TARGET_REPLACE_ITER = 1000
GAMMA = 0.9
MEM_SIZE = 2000

class Agent():
    def __init__(self, env) -> None:
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.max_mem = MEM_SIZE
        self.env = env
        self.eval, self.target = Net(n_states, n_actions), Net(n_states, n_actions)
        self.optimizer = optim.Adam(self.eval.parameters(), lr=LR)
        self.buffer = ReplayBuffer(max_size=self.max_mem,input_shape=[n_states], n_actions=n_actions)
        self.loss_func = nn.MSELoss()

        self.mem_count = 0
        self.learn_steps = 0

    def store(self, state, action, state_, rewards, done):
        self.mem_count += 1
        self.buffer.store(state, action, state_, rewards, done)

    def choose_action(self, state):
        if np.random.uniform() > 0.95:
            return self.env.action_space.sample()

        state = torch.tensor(state, dtype=torch.float).to(self.eval.device)
        action_value = self.eval(state)
        action = torch.argmax(action_value).cpu().data.item()
        return action
    
    def learn(self):
        if self.learn_steps % TARGET_REPLACE_ITER == 0:
            self.target.load_state_dict(self.eval.state_dict())
        self.learn_steps += 1

        states, actions, rewards, states_, dones = self.buffer.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float).to(self.eval.device)
        actions = torch.tensor(actions, dtype=torch.int64)[:, 0].resize(BATCH_SIZE, 1).to(self.eval.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.eval.device)
        rewards = torch.tensor(rewards, dtype=torch.float).reshape(BATCH_SIZE, 1).to(self.eval.device)

        q_values = self.eval(states).gather(1, actions)
        q_next = self.eval(states_).detach()
        q_target = rewards + GAMMA*torch.max(q_next, 1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_values, q_target)

        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step() 
        
