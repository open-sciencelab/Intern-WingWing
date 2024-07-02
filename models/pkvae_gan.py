import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-7

class ConditionalBatchNormalization2D(nn.Module):
    def __init__(self, channels):
        super(ConditionalBatchNormalization2D, self).__init__()
        
        self.batch_norm = nn.BatchNorm2d(channels, affine=False)
        self.gamma_dense = nn.Linear(channels, channels)
        self.beta_dense = nn.Linear(channels, channels)
        
    def forward(self, inputs, condition):
        x = self.batch_norm(inputs)
        
        gamma = self.gamma_dense(condition)
        beta = self.beta_dense(condition)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        x = x + x * gamma + beta
        
        return x

class SNLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features, 
                                bias=True)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)
    
    def forward(self, x):
        return self.linear(x)

class SNConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size = 1,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
        dilation = 1, groups = 1, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight.data, 1.)
        
    def forward(self, input):
        return self.conv(input)

class CDiscriminator(nn.Module):
    def __init__(self, latent_dim = 3, n_points = 192, cond_dim = 11):
        super(CDiscriminator, self).__init__()
        def Conv(in_channel, out_channel):
            layer = []
            layer.append(SNConv2d(in_channel, out_channel, kernel_size=self.kernel_size, stride=(1, 2), padding=(1,1)))
            # layer.append(nn.BatchNorm2d(out_channel, momentum=0.9))
            layer.append(nn.LeakyReLU(0.2))
            layer.append(nn.Dropout(self.dropout))
            return layer

        self.depth = 64
        self.dropout = 0.4
        self.kernel_size = (3, 4)
        self.n_point = n_points
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        layer = Conv(1, self.depth)
        for i in range(5):
            ii = pow(2, i)
            layer += Conv(self.depth * ii, self.depth * ii * 2)
        self.Conv = nn.Sequential(
            *layer
        )
        self.Dense = nn.Sequential(
            nn.Flatten(),
            SNLinear(self.depth * pow(2, 5) * int(self.n_point * 2/64), 1024),
            # nn.BatchNorm1d(1024, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.cond_dense = nn.Linear(self.cond_dim, 1024)
        self.cond_LReLU = nn.LeakyReLU(0.2)
        self.cond_bn = nn.BatchNorm1d(1024, momentum=0.9)
        self.dense_d = SNLinear(1024, 1)
        self.dense_q = nn.Sequential(
            SNLinear(1024, 128),
            # nn.BatchNorm1d(128, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.dense_q_mean = SNLinear(128, self.latent_dim)
        self.dense_q_logstd = SNLinear(128, self.latent_dim)

    def forward(self, x, cond):
        # x:[bz, n_points, 2, 1]
        x = torch.transpose(x, 1, 3)
        cond = self.cond_bn(self.cond_LReLU(self.cond_dense(cond)))
        # x:[bz, 1, 2, n_points]
        x = self.Conv(x)
        x = self.Dense(x)
        cond = torch.sum(cond * x, dim=-1, keepdim=True)

        d = self.dense_d(x) + cond
        q = self.dense_q(x)
        q_mean = self.dense_q_mean(q)
        q_logstd = self.dense_q_logstd(q)
        q_logstd = torch.maximum(q_logstd, -16 * torch.ones_like(q_logstd))
        q_mean = q_mean.view(-1, 1, self.latent_dim)
        q_logstd = q_logstd.view(-1, 1, self.latent_dim)
        q = torch.cat([q_mean, q_logstd], dim=1)

        return d, q

class PKVAE(nn.Module):
    def __init__(self, feature_size=257*2, latent_size=10, condition_size=37):
        super(PKVAE, self).__init__()
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

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size//2,2,1)
        c: (bs, class_size,1)
        '''
        bs = x.shape[0]
        x = x.reshape(bs,-1)
        y =  torch.cat([x,c],dim=1) # (bs,feature_size+condition_size)
        h1 = self.encoder_fc(y)
        z_mu = self.fc_mean(h1)
        z_var = F.softplus(self.fc_var(h1))
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, condition_size)
        '''
        y = torch.cat([z,c],dim=1)   
        return self.decoder_fc(y)
    
    def sample(self,c):
        '''
        z: (bs, latent_size)
        c: (bs, condition_size, 1)
        '''
        batch = c.shape[0]
        z = torch.randn((batch,self.latent_size)).to(c.device)
        # import pdb; pdb.set_trace()
        recons_batch = self.decode(z, c).reshape(-1,self.feature_size//2,2,1)
        return recons_batch
    
    def forward(self, x, c):
        # x: (bs, feature_size//2,2,1)
        # c: (bs, condition_size)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recons_batch = self.decode(z, c).reshape(-1,self.feature_size//2,2,1)
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