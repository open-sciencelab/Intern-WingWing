import torch
import torch.nn as nn


class ConvBnSiLu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super().__init__()
        self.module=nn.Sequential(nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                                  nn.BatchNorm1d(out_channels),
                                  nn.SiLU(inplace=True))
    def forward(self,x):
        return self.module(x)

class ResidualBottleneck(nn.Module):
    '''
    shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.branch1=nn.Sequential(nn.Conv1d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm1d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels//2,in_channels//2,1,1,0),
                                    nn.Conv1d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm1d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))

    def forward(self,x):
        x1,x2=x.chunk(2,dim=1)
        x=torch.cat([self.branch1(x1),self.branch2(x2)],dim=1)

        return x

class ResidualDownsample(nn.Module):
    '''
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch1=nn.Sequential(nn.Conv1d(in_channels,in_channels,3,2,1,groups=in_channels),
                                    nn.BatchNorm1d(in_channels),
                                    ConvBnSiLu(in_channels,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels,out_channels//2,1,1,0),
                                    nn.Conv1d(out_channels//2,out_channels//2,3,2,1,groups=out_channels//2),
                                    nn.BatchNorm1d(out_channels//2),
                                    ConvBnSiLu(out_channels//2,out_channels//2,1,1,0))

    def forward(self,x):
        x=torch.cat([self.branch1(x),self.branch2(x)],dim=1)

        return x

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()
    def forward(self,x,t):
        t_emb=self.mlp(t).unsqueeze(-1)
        x=x+t_emb
  
        return self.act(x)
    
class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,out_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=out_channels//2)
        self.conv1=ResidualDownsample(out_channels//2,out_channels)
    
    def forward(self,x,t=None):
        x_shortcut=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x_shortcut,t)
        x=self.conv1(x)

        return [x,x_shortcut]
        
class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='linear',align_corners=False)
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,in_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=in_channels,out_dim=in_channels//2)
        self.conv1=ResidualBottleneck(in_channels//2,out_channels//2)

    def forward(self,x,x_shortcut,t=None):
        x=self.upsample(x)
        x=torch.cat([x,x_shortcut],dim=1)
        x=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x,t)
        x=self.conv1(x)

        return x        
    


class Unet(nn.Module):
    '''
    Modified UNet design to include conditioned inputs.
    '''
    def __init__(self, timesteps, time_embedding_dim, in_channels=1, out_channels=1, base_dim=32, dim_mults=[2, 4, 8, 16], cond_dim=2):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        assert base_dim % 2 == 0

        channels = self._cal_channels(base_dim, dim_mults)

        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.cond_fc = nn.Linear(cond_dim, time_embedding_dim)  # New layer for condition embedding

        self.encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(c[1], c[0], time_embedding_dim) for c in channels[::-1]])

        self.mid_block = nn.Sequential(*[ResidualBottleneck(channels[-1][1], channels[-1][1]) for i in range(2)],
                                       ResidualBottleneck(channels[-1][1], channels[-1][1]//2))

        self.final_conv = nn.Conv1d(channels[0][0]//2, out_channels, kernel_size=1)
        self.act = nn.Tanh()

    def forward(self, x, t, cond):
        # Process condition input
        cond_emb = self.cond_fc(cond)
        #print("t type before embedding:",t.type())
        t_emb = self.time_embedding(t)
        combined_emb = t_emb + cond_emb  # Combine embeddings
        x = self.init_conv(x)
        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, combined_emb)  # Pass the combined embedding to each block
            encoder_shortcuts.append(x_shortcut)
        x = self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block, shortcut in zip(self.decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, shortcut, combined_emb)  # Pass the combined embedding to each block
        x = self.final_conv(x)

        return x

    def _cal_channels(self,base_dim,dim_mults):
        dims=[base_dim*x for x in dim_mults]
        dims.insert(0,base_dim)
        channels=[]
        for i in range(len(dims)-1):
            channels.append((dims[i],dims[i+1])) # in_channel, out_channel

        return channels
    
class ConditionedUnet(Unet):
    def __init__(self, timesteps, time_embedding_dim, in_channels, out_channels, base_dim, dim_mults, cond_dim):
        super().__init__(timesteps, time_embedding_dim, in_channels, out_channels, base_dim, dim_mults)
        # Additional layers for conditioning
        self.cond_fc = nn.Linear(cond_dim, time_embedding_dim)  # Map Cl/Cd to a suitable size

    def forward(self, x, t, cond):
        cond_emb = self.cond_fc(cond)  # Get condition embedding
        t_emb = self.time_embedding(t)
        t_emb = t_emb + cond_emb  # Combine time and condition embeddings
        return super().forward(x, t_emb)  # Call the original Unet forward


if __name__=="__main__":
    x=torch.randn(128,1,32)
    t=torch.randint(0,1000,(128,))
    model=Unet(1000,128)
    y=model(x,t)
    print(y.shape)