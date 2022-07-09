from turtle import forward
import torch as T
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, n_input=28*28, lr=0.005) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, n_input),
            nn.Sigmoid()
        )

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.lose_func = nn.MSELoss()


    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded