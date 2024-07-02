import torch
from torch import nn

'''https://arxiv.org/abs/2004.12585'''
class BN_Layer(nn.Module):
    def __init__(self, dim_z, tau=0.5, mu=True):
        super(BN_Layer, self).__init__()
        self.dim_z = dim_z

        self.tau = torch.tensor(tau)  # tau: float in range (0,1)
        self.theta = torch.tensor(0.5, requires_grad=True)

        self.gamma1 = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.theta))  # for mu
        self.gamma2 = torch.sqrt((1 - self.tau) * torch.sigmoid((-1) * self.theta))  # for var

        self.bn = nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = True

        if mu:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma1)
        else:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma2)

    def forward(self, x):  # x:(batch_size,dim_z)
        x = self.bn(x)
        return x


class VAE(nn.Module):
    def __init__(self, feature_size=257, latent_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.feature_size = feature_size

        # encode
        self.fc1  = nn.Linear(feature_size, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc21 = nn.Linear(128, latent_size)
        self.fc22 = nn.Linear(128, latent_size)
        # self.bn_mu = BN_Layer(latent_size, mu=True)
        # self.bn_var = BN_Layer(latent_size, mu=False)

        # decode
        self.fc3 = nn.Linear(latent_size, 2048)
        self.fc4 = nn.Linear(2048, 128)
        self.fc5 = nn.Linear(128, feature_size)

    def encode(self, x):  
        h1 = torch.relu(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        z_mu = self.fc21(h2)
        z_logvar = self.fc22(h2)
        # z_mu = self.bn_mu(z_mu)
        # z_var = self.bn_var(z_var)
        return z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = torch.relu(self.fc3(z))
        z = torch.relu(self.fc4(z))
        x = torch.tanh(self.fc5(z))
        return x

    def forward(self, x):
        x = x.reshape(-1, self.feature_size) # [B, 1, N]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VQ_VAE(nn.Module):
    def __init__(self, feature_size=257*2, latent_size=128, num_embeddings=512, embedding_dim=128):
        super(VQ_VAE, self).__init__()
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # encode
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ELU(),
            nn.Linear(512, latent_size)
        )
        
        # codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # decode
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ELU(),
            nn.Linear(512, feature_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, self.feature_size))  # Get latent representation
        z_quantized, indices = self.vector_quantization(z)  # Quantize the latent space
        x_recon = self.decoder(z_quantized)  # Decode the quantized latent space
        return x_recon, z, z_quantized

    def vector_quantization(self, z):
        distances = torch.cdist(z, self.codebook.weight, p=2.0)  # Compute distances
        indices = torch.argmin(distances, dim=1)  # Find nearest neighbors in the codebook
        z_quantized = self.codebook(indices)  # Quantized latent space
        return z_quantized, indices


if __name__ == '__main__':
    model = VQ_VAE()
    x = torch.randn(10, 257*2)
    x_recon, z, z_quantized = model(x)
    print(x_recon.shape, z.shape, z_quantized.shape)