import torch
from torch import nn
import torch.nn.functional as F
 

class EK_VAE(nn.Module):
    def __init__(self, feature_size=11, latent_size=10, condition_size=37):
        super(EK_VAE, self).__init__()
        self.feature_size = feature_size
        self.condition_size = condition_size
        self.latent_size = latent_size
 
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.feature_size+self.condition_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            )

        self.fc_mean = nn.Linear(16, self.latent_size)
        self.fc_var = nn.Linear(16, self.latent_size)

   
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_size+condition_size, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, self.feature_size),
            # nn.Tanh()
            )

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): 
        inputs =  torch.cat([x,c],dim=1)  
        y = inputs.squeeze(dim=-1)
        h1 = self.encoder_fc(y)
        z_mu = self.fc_mean(h1)
        z_var = F.softplus(self.fc_var(h1))
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