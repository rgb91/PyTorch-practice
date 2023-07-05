import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img_to_hid = nn.Linear(input_dim, h_dim)
        self.hid_to_mu = nn.Linear(h_dim, z_dim)
        self.hid_to_sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_to_hid = nn.Linear(z_dim, h_dim)
        self.hid_to_img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_to_hid(x))
        mu = self.hid_to_mu(h)
        sigma = self.hid_to_sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_to_hid(z))
        img = self.hid_to_img(h)
        return torch.sigmoid(img)

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparam = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparam)

        # x_reconstructed for reconstruction loss
        # mu & sigma for decay and divergence
        return x_reconstructed, mu, sigma


if __name__ == '__main__':
    x = torch.randn(1, 28*28)
    vae = VariationalAutoEncoder(input_dim=784, h_dim=200, z_dim=20)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape, mu.shape, sigma.shape)

