import torch
import torch.nn as nn
import torch.nn.functional as F


class CDiscriminator(nn.Module):
    def __init__(self, feature_size=257, condition_size=37):
        super(CDiscriminator, self).__init__()
        self.feature_size = feature_size
        self.condition_size = condition_size
        self.label_embedding = nn.Linear(condition_size,condition_size)

        self.dis = nn.Sequential(
            nn.Linear(self.feature_size + self.condition_size, 256),   
            nn.LeakyReLU(0.2),   
            nn.Linear(256, 256),   
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        x = x.squeeze(dim=-1)
        c = self.label_embedding(c.squeeze(dim=-1))
        x = torch.concat([x,c],dim=1)
        x = self.dis(x)
        return x

    

class CGenerator(nn.Module):
    def __init__(self, latent_size=10, condition_size=37, feature_size=257):
        super(CGenerator, self).__init__()
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.feature_size = feature_size

        self.label_embedding = nn.Linear(condition_size,condition_size)

        self.gen = nn.Sequential(
            nn.Linear(latent_size+condition_size, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, feature_size),
            # nn.Tanh()
        )
    
    def sample(self,c):
        batch = c.shape[0]
        z = torch.randn((batch,self.latent_size)).to(c.device)
        recons_batch = self(z,c)
        return recons_batch
    
    def forward(self, x, c):
        c = self.label_embedding(c.squeeze(dim=-1))
        x = torch.concat([x,c],dim=1)
        x = self.gen(x)
        x = x.reshape(-1, self.feature_size, 1)
        return x