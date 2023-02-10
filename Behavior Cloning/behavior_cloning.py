import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
from gym.wrappers import Monitor
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
import imageio

BUFFER_SIZE = 10000
BATCH_SIZE = 128
NUM_EPOCH = 100000
env_id = "Hopper-v3"

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
        return self.out(state)

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

writter = SummaryWriter()
env = gym.make(env_id)
# video_env =  Monitor(gym.make("HalfCheetah-v2"), './video', force=True)
'''
model_name = "PPO"
expert_model = PPO("MlpPolicy", env, verbose=0)

'''
model_name = "TD3"
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
expert_model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0, device='cuda:0', train_freq=1, learning_rate=3*(10**-4), gradient_steps=1, batch_size=256)
expert_model = expert_model.load('assets/experts/{}_{}_best.zip'.format(model_name, env_id))

actor = Actor(n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
db = Replay_Buffer(buffer_size=BUFFER_SIZE, n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])

print('Start Collecting Data')
while db.counter <= BUFFER_SIZE:
    obs = env.reset()
    done = False
    total_rewards = 0
    frames = []
    frames.append(env.render(mode='rgb_array'))
    while not done:
        action, _state = expert_model.predict(obs, deterministic=True)
        db.store(obs, action)
        obs, reward, done, info = env.step(action)
        total_rewards += reward
        frames.append(env.render(mode='rgb_array'))

imageio.mimsave('assets/expert_{}_{}.gif'.format(env_id, model_name),[np.array(frame) for i, frame in enumerate(frames) if i%2 ==0], fps=20)
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
        frames = []
        frames.append(env.render(mode='rgb_array'))
        while not done:
            action = actor(obs).cpu().detach().numpy()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            frames.append(env.render(mode='rgb_array'))
        print(total_reward, running_loss/1000)
        writter.add_scalar('BC/average_loss', loss, i)
        writter.add_scalar('BC/rewards', total_reward, i)
        running_loss = 0.0

imageio.mimsave('assets/learner_{}.gif'.format(env_id),[np.array(frame) for i, frame in enumerate(frames) if i%2 ==0], fps=20)


