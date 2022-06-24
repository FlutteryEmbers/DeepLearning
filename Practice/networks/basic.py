from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.output = nn.Linear(128, n_outputs)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        output = self.output(F.relu(x))
        return output
