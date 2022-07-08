import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as Normal

class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions, max_action, lr=1e-3, fc1_dims=256, fc2_dims=256, reparam_noise=1e-6) -> None:
        super(Actor, self).__init__()
        self.reparam_noise = reparam_noise
        self.max_action = max_action

        self.fc1 = nn.Linear(*n_inputs, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)

        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def foward(self, state):
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        action_dist = Normal(mu, sigma)

        if reparameterize:
            actions = action_dist.rsample()
        else:
            actions = action_dist.sample()

        action = T.tanh(actions) * T.tensor(self.max_actions).to(self.device)

        log_prob = action_dist.log_prob(actions)
        log_prob -= T.log(1-action.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Value(nn.Module):
    def __init__(self, n_inputs, lr=1e-3, fc1_dims=256, fc2_dims=256) -> None:
        super(Value, self).__init__()
        self.fc1 = nn.Linear(*n_inputs, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        state_value = self.v(x)

        return state_value

class Critic(nn.Module):
    def __init__(self, n_inputs, n_actions, lr=1e-3, fc1_dims=256, fc2_dims=256) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(*n_inputs + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, actions):
        state = state.to(self.device)
        actions = actions.to(self.device)

        x = F.relu(self.fc1(T.cat([state, actions], dim=1)))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)

        return q_value
