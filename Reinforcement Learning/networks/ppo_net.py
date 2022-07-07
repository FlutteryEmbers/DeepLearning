import torch as T
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions, fc1_dim=256, fc2_dim=256, lr = 1e-3) -> None:
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(*n_inputs, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        output = self.network(state)

        pi = T.distributions.Categorical(output)
        return pi

class Critic(nn.Module):
    def __init__(self, n_inputs, fc1_dims=256, fc2_dims=256, lr = 1e-4) -> None:
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(*n_inputs, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        state = state.to(self.device)

        return self.network(state)
