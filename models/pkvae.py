import torch
from torch import nn
from torch.nn import functional as F


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

class PK_VAE(nn.Module):
    def __init__(self, feature_size=257, latent_size=20, condition_size=11+26):
        super(PK_VAE, self).__init__()
        self.feature_size = feature_size
        self.condition_size = condition_size
        self.latent_size = latent_size

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.feature_size+self.condition_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            )

        self.fc_mean = nn.Linear(64, self.latent_size)
        self.fc_var = nn.Linear(64, self.latent_size)
        self.bn_mu = BN_Layer(self.latent_size, tau=0.5, mu=True)
        self.bn_var = BN_Layer(self.latent_size, tau=0.5, mu=False)

   
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_size+condition_size, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, self.feature_size),
            # nn.Tanh()
            )

        self.elu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): 
        inputs =  torch.cat([x,c],dim=1)  
        y = inputs.squeeze(dim=-1)
        h1 = self.encoder_fc(y)
        z_mu = self.bn_mu(self.fc_mean(h1))
        z_var = F.softplus(self.bn_var(self.fc_var(h1)))
        z_mu = self.fc_mean(h1)
        z_var = self.elu(self.fc_var(h1))
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): 
        c = c.squeeze(dim=-1)
        y = torch.cat([z,c],dim=1)   
        return self.decoder_fc(y)
    
    def sample(self,c):
        batch = c.shape[0]
        z = torch.randn((batch,self.latent_size)).to(c.device)
        recons_batch = self.decode(z, c).reshape(-1,self.feature_size,1)
        return recons_batch
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recons_batch = self.decode(z, c).reshape(-1,self.feature_size,1)
        return recons_batch, mu, logvar
  
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                nn.init.xavier_normal_(m.weight.data,validate_args=False)
                m.bias.data.zero_(validate_args=False)