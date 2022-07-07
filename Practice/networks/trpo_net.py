import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions, fc_dim1 = 64, fc_dim2 = 64, lr = 1e-3) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_inputs, fc_dim1)
        self.fc2 = nn.Linear(fc_dim1, fc_dim2)
        self.output = nn.Linear(fc_dim2, n_actions)

        self.output.weight.data.mul_