from turtle import forward
from importlib_metadata import distribution
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, n_inputs, lr) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(*n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)
        self.optim = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions, lr) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(*n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.optim = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(F.relu(self.fc2(x)))
        pi = T.distributions.Categorical(x)

        return pi