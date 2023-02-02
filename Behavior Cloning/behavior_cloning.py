import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import gym
from stable_baselines3 import PPO

BUFFER_SIZE = 50000
BATCH_SIZE = 64
NUM_EPOCH = 1000000

class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_actions)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        action = F.relu(self.out(state))
        return action

class Replay_Buffer():
    def __init__(self, buffer_size, n_inputs, n_actions) -> None:
        self.buffer_size = buffer_size
        self.state_memory = np.zeros((buffer_size, n_inputs), dtype=np.float32)
        self.action_memory = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self.counter = 0

    def store(self, state, action):
        idx = self.counter % self.buffer_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.counter += 1

    def sample(self, batch_size):
        size = min(self.counter, self.buffer_size)
        batch = np.random.choice(size, batch_size)
        return self.state_memory[batch], self.action_memory[batch]

env = gym.make("HalfCheetah-v2")
expert_model = PPO("MlpPolicy", env, verbose=0)
expert_model = expert_model.load('assets/experts/ppo_best.zip')

actor = Actor(n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
db = Replay_Buffer(buffer_size=BUFFER_SIZE, n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])

print('Start Collecting Data')
while db.counter <= BUFFER_SIZE:
    obs = env.reset()
    done = False
    total_rewards = 0
    while not done:
        action, _state = expert_model.predict(obs, deterministic=True)
        db.store(obs, action)
        obs, reward, done, info = env.step(action)
        total_rewards += reward

print(total_rewards)

print('Start Trainning')
# Train Model
running_loss = 0
loss_fn = nn.MSELoss()

for i in range(NUM_EPOCH):
    state, action = db.sample(BATCH_SIZE)

    actor.optimizer.zero_grad()
    pred_action = actor(state)
    action = torch.tensor(action).to(actor.device)
    loss = loss_fn(pred_action, action)
    loss.backward()
    actor.optimizer.step()
    running_loss += loss.item()
    if i % 1000 == 0:
        total_reward = 0
        obs = env.reset()
        done = False
        while not done:
            action = actor(obs).cpu().detach().numpy()
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(total_reward, running_loss/1000)
        running_loss = 0.0

