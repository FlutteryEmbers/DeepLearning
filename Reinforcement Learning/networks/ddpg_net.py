import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions, lr = 1e-4, fc1_dims = 400, fc2_dims = 300) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(*n_inputs, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)

        # initialize the weight
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc2.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 3e-3
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)


        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        output = T.tanh(self.fc3(x))

        return output

class Critic(nn.Module):
    def __init__(self, n_inputs, n_actions, lr = 1e-3, fc1_dims = 400, fc2_dims = 300, weight_decay = 1e-2) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(*n_inputs, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.action_value = nn.Linear(n_actions, fc2_dims)

        self.q = nn.Linear(fc2_dims, 1)

        # initialize layer weights and bias 
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay=weight_decay)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        state_value = F.relu(self.bn1(self.fc1(state)))

        state_value = self.bn2(self.fc2(state_value))
        action_value = F.relu(self.action_value(action))

        state_action_value = F.relu(T.add(state_value, action_value))
        q = self.q(state_action_value)

        return q