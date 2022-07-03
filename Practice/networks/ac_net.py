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
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        pi = T.distributions.Categorical(x)

        return pi

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, lr) -> None:
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*n_inputs, 2048)
        self.fc2 = nn.Linear(2048, 1536)
        self.pi = nn.Linear(1536, n_actions)
        self.value = nn.Linear(1536, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        pi = self.pi(x)
        value = self.value(x)

        return (pi, value)
