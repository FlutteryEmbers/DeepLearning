import torch as T
import torch.nn as nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self, n_input=28*28, lr=0.005) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 12),
            nn.LeakyReLU(0.1),
            nn.Linear(12, 4),
        )

        self.mu = nn.Linear(4, 2)
        self.log_var = nn.Linear(4, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.LeakyReLU(0.1),
            nn.Linear(12, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, n_input),
            nn.Sigmoid()
        )

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def reparameterize(self, z_mu, z_log_var):
        eps = T.randn(z_mu.size[0], z_mu.size[1],).to(z_mu.get_device())
        z = z_mu + eps * T.exp(z_log_var/2.)
        return z

    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        mu, log_var = self.mu(encoded), self.log_var(encoded)
        encoded = self.reparameterize(mu, log_var)
        decoded = self.decoder(encoded)
        return encoded, mu, log_var, decoded