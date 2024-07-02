import torch
import torch.nn as  nn
import numpy as np
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

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

class SNConvTranspose(nn.Module):
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
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
        dilation = 1, groups = 1, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight.data, 1.)
        
    def forward(self, input):
        return self.conv(input)


class Res_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Block, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(out_channel, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(out_channel, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.extra = nn.Sequential()
        if in_channel != out_channel:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1)),
                nn.BatchNorm2d(out_channel, momentum=0.9)
            )
        self.Relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.Conv(x)
        x = self.extra(x)
        out = self.Relu(out + x)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=1),
            nn.BatchNorm2d(out_channel, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.Conv_x = nn.Sequential(
            nn.Conv2d(in_channel, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1024, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.blk1 = Res_Block(out_channel, 128)
        self.blk2 = Res_Block(128, 256)
        self.blk3 = Res_Block(256, 512)
        self.blk4 = Res_Block(512, 1024)
        self.out = nn.Sequential(
            nn.Conv2d(1024, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channel, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.Relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.Conv(x)
        x = self.Conv_x(x)
        out = self.blk4(self.blk3(self.blk2(self.blk1(out))))
        out = self.Relu(x + out)
        out = self.out(out)
        return out

class CGenerator(nn.Module):
    def __init__(self, latent_dim, noise_dim, cond_dim,  n_points = 192, bezier_degree = 31, bounds = (0.0, 1.0)):
        super(CGenerator, self).__init__()

        def conv2d_transpose(input_c, output_c):
            layer = []
            layer.append(nn.ConvTranspose2d(input_c, output_c, kernel_size=self.kernel_size, stride=(1, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_c, momentum = 0.9))
            layer.append(nn.LeakyReLU(0.2))
            return layer

        def Dense(input_c, output_c):
            layer = []
            layer.append(nn.Linear(input_c, output_c))
            layer.append(nn.BatchNorm1d(output_c, momentum = 0.9))
            layer.append(nn.LeakyReLU(0.2))
            return layer

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.X_shape = (n_points, 2, 1)
        self.bezier_degree = bezier_degree
        self.bounds = bounds
        self.depth_cpw = 32 * 8
        self.dim_cpw = int((self.bezier_degree+1)/8)
        self.kernel_size = (3, 4)

        self.Dense = nn.Sequential(
            *Dense(self.latent_dim + self.noise_dim, 1024),
            *Dense(1024, self.dim_cpw * 3 * self.depth_cpw)
        )

        self.condDense1 = nn.Sequential(
            *Dense(self.cond_dim, int(self.depth_cpw/8))
        )
        self.condDense2 = nn.Sequential(
            *Dense(int(self.depth_cpw/8), int(self.depth_cpw/4))
        )
        self.condDense3 = nn.Sequential(
            *Dense(int(self.depth_cpw/4), int(self.depth_cpw/2))
        )

        self.conv2d_transpose = nn.Sequential(
            *conv2d_transpose(self.depth_cpw, int(self.depth_cpw/2)),
            *conv2d_transpose(int(self.depth_cpw/2), int(self.depth_cpw/4)),
            *conv2d_transpose(int(self.depth_cpw/4), int(self.depth_cpw/8)),
        )
        self.batchnorm1 = ConditionalBatchNormalization2D(int(self.depth_cpw/8))
        self.batchnorm2 = ConditionalBatchNormalization2D(int(self.depth_cpw/4))
        self.batchnorm3 = ConditionalBatchNormalization2D(int(self.depth_cpw/2))


        self.Res1 = ResNet(int(self.depth_cpw/8), int(self.depth_cpw/4))
        self.Res2 = ResNet(int(self.depth_cpw/4), int(self.depth_cpw/2))
        self.Res3 = ResNet(int(self.depth_cpw/2), self.depth_cpw)
        self.Conv_cpw = nn.Sequential(
            nn.Conv2d(int(self.depth_cpw/8), self.depth_cpw, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(self.depth_cpw, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.cpw_out = nn.Sequential(
            nn.Conv2d(self.depth_cpw, int(self.depth_cpw/8), kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(int(self.depth_cpw/8), momentum=0.9),
            nn.LeakyReLU(0.2)
        )

        self.Conv_cp = nn.Sequential(
            nn.Conv2d(int(self.depth_cpw/8), 1, (2,1), padding='valid'),
            nn.Tanh()
        )
        self.Conv_w = nn.Sequential(
            nn.Conv2d(int(self.depth_cpw/8), 1, (3,1), padding='valid'),
            nn.Sigmoid()
        )
        self.Dense_db = nn.Sequential(
            *Dense(self.latent_dim + self.noise_dim ,1024),
            *Dense(1024, 256),
            nn.Linear(256, self.X_shape[0]-1),
            nn.Softmax()
        )

    def forward(self, c, z, cond):
        cz = torch.cat([c, z], dim=-1)
        cpw = self.Dense(cz)
        cpw = torch.reshape(cpw, ((-1, self.depth_cpw, 3, self.dim_cpw)))
        cpw = self.conv2d_transpose(cpw) # [bz, 32, 3, 32]
        tmp = cpw
        cond = self.condDense1(cond) # [bz, 32]
        cpw = self.batchnorm1(cpw,cond)
        cpw = self.Res1(cpw)

        cond = self.condDense2(cond)
        cpw = self.batchnorm2(cpw,cond)
        cpw = self.Res2(cpw)
        cond = self.condDense3(cond)
        cpw = self.batchnorm3(cpw,cond)
        out = self.Res3(cpw)

        # A Resnet block
        cpw = self.Conv_cpw(tmp)
        cpw = self.cpw_out(out + cpw)

        cp = self.Conv_cp(cpw)
        cp = torch.squeeze(cp, dim=-3)
        w = self.Conv_w(cpw)
        w = torch.squeeze(w, dim=-3)
        db = self.Dense_db(cz)
        ub = F.pad(db, (1,0,0,0), "constant", value=0)
        ub = torch.cumsum(ub, dim=1)
        ub = torch.minimum(ub, torch.ones_like(ub))
        ub = torch.unsqueeze(ub, dim=-1)
        # print(ub.size())
        num_control_points = self.bezier_degree + 1
        lbs = torch.tile(ub, (1, 1, num_control_points))
        pw1 = torch.arange(0, num_control_points, dtype=torch.float32).to(device)
        pw1 = torch.reshape(pw1, (1, 1, -1))
        pw2 = torch.flip(pw1, dims=[-1])
        lbs = torch.add(torch.multiply(pw1, torch.log(lbs + EPSILON)), torch.multiply(pw2, torch.log(1 - lbs + EPSILON)))
        # lbs = torch.add(torch.multiply(pw1, torch.log(lbs + EPSILON)), torch.multiply(pw2, torch.log(1-lbs+EPSILON)))
        lc = torch.add(torch.lgamma(pw1+1), torch.lgamma(pw2+1))
        lc = torch.subtract(torch.lgamma(torch.tensor(num_control_points).float().to(device)), lc)
        lbs = torch.add(lbs, lc)
        bs = torch.exp(lbs)
        cp_w = torch.multiply(cp, w)
        cp_w = torch.transpose(cp_w, 1, 2)
        dp = torch.matmul(bs, cp_w)
        cp = torch.transpose(cp, 1, 2)
        w = torch.transpose(w, 1, 2)
        bs_w = torch.matmul(bs, w)
        dp = torch.div(dp, bs_w)
        dp = torch.unsqueeze(dp, dim=-1)

        return dp, cp, w, ub, db

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
        x = torch.transpose(x, 1, 3)
        cond = self.cond_bn(self.cond_LReLU(self.cond_dense(cond)))
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


 