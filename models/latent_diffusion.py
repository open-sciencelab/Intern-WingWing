import torch.nn as nn
import torch
import math
from tqdm import tqdm
import torch
import torch.nn as nn

class AirfoilDiffusion(nn.Module):
    def __init__(self,model,latent_size,in_channels,out_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8], device="cuda"):
        super().__init__()
    
        self.latent_size=latent_size
        self.in_channels=in_channels
        self.timesteps=timesteps

    

        self.beta=self._cosine_variance_schedule(timesteps).to(device)
        self.alpha=(1.0-self.beta).to(device)
        self.alpha_bar=torch.cumprod(self.alpha,dim=-1).to(device)
        self.alpha_prev_bar=torch.cat([torch.tensor([1.0],device=self.alpha.device),self.alpha_bar[:-1]],dim=0)
        self.sqrt_alpha_bar=torch.sqrt(self.alpha_bar).to(device)
        self.sqrt_one_minus_alpha_bar=torch.sqrt(1.0-self.alpha_bar).to(device)


        self.model=model
        self.cond_embedder = nn.Linear(37, 2)

    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas
        

    def forward(self, x, noise, y1, y2, reduced_cond):
        # Combine Cl and Cd into a single tensor
        cond = torch.cat((y1, y2), dim = -1)
        # cond = self.cond_embedder(cond)
        target_time = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)

        x_t = self._forward_diffusion(x, target_time, noise, reduced_cond)
        
        pred = self.model(x_t, target_time, cond)
        return pred

    
    
    def _forward_diffusion(self, x_0, t, noise, cond):
        Cl = cond[:, 0].unsqueeze(1).unsqueeze(2)  # Shape becomes [batch_size, 1, 1] compatible with x_0
        Cd = cond[:, 1].unsqueeze(1).unsqueeze(2)  # Same as above
        
        alpha_scaled = self.sqrt_alpha_bar[t].unsqueeze(1).unsqueeze(2)  # Ensure shape is [batch_size, 1, 1]
        one_minus_alpha_scaled = self.sqrt_one_minus_alpha_bar[t].unsqueeze(1).unsqueeze(2)  # Same as above

        q = alpha_scaled * x_0 * (1 + 0.1 * Cl) + one_minus_alpha_scaled * noise * (1 + 0.1 * Cd)

        return q


    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise, cl_cd):
        alpha_t = self.alpha[t]
        beta_t = self.beta[t]
        sqrt_one_minus_alpha_t_cumprod = self.sqrt_one_minus_alpha_bar[t]

        # Pass cl_cd as part of the model's prediction process
        prediction = self.model(x_t, t, cl_cd)  
        #print("prediction shape in reverse diffusion: ", prediction.shape)

        a = (1 / torch.sqrt(alpha_t)).reshape(x_t.shape[0], 1, 1)
        beta_t_reshaped = beta_t.view(-1, 1, 1)
        #print("beta_t_reshaped shape in reverse diffusion: ", beta_t_reshaped.shape)
        sqrt_one_minus_alpha_t_cumprod_reshaped = sqrt_one_minus_alpha_t_cumprod.view(-1, 1, 1)
        #print("sqrt_one_minus_alpha_t_cumprod_reshaped shape in reverse diffusion: ", sqrt_one_minus_alpha_t_cumprod_reshaped.shape)

        b = (x_t - (beta_t_reshaped / sqrt_one_minus_alpha_t_cumprod_reshaped) * prediction)
        #print("b shape in reverse diffusion: ", b.shape)
        mu = a * b
       #print("mu shape in reverse diffusion: ", mu.shape)

        if t.min() > 0:
            sigma = torch.sqrt(beta_t_reshaped)
        else:
            sigma = torch.tensor(0.0).to(x_t.device)

        return mu + sigma * noise
    
    @torch.no_grad()
    def _reverse_diffusion_x(self, x_t, t, noise, cl_cd):
        alpha_t = self.alpha[t].view(-1,1,1)
        beta_t = self.beta[t].view(-1,1,1)
        sqrt_alpha_bar_t_minus_1 = self.sqrt_alpha_bar[t-1].view(-1, 1, 1)
        sqrt_one_minus_alpha_t_cumprod = self.sqrt_one_minus_alpha_bar[t].view(-1,1,1)
        sqrt_one_minus_alpha_t_minus_1_cumprod = self.sqrt_one_minus_alpha_bar[t-1].view(-1,1,1)
        one_minus_alpha_t_cumprod = (sqrt_one_minus_alpha_t_cumprod * sqrt_one_minus_alpha_t_cumprod).view(-1, 1, 1)
        one_minus_alpha_t_minus_1_cumprod = (sqrt_one_minus_alpha_t_minus_1_cumprod * sqrt_one_minus_alpha_t_minus_1_cumprod).view(-1, 1, 1)
        
        x_0 = self.model(x_t, t, cl_cd)  

        a = sqrt_alpha_bar_t_minus_1 * beta_t / (one_minus_alpha_t_cumprod)
        a = a.view(-1, 1, 1)

        b = (torch.sqrt(alpha_t)).reshape(x_t.shape[0], 1, 1) * (one_minus_alpha_t_minus_1_cumprod) / one_minus_alpha_t_cumprod
        b = b.view(-1, 1, 1)

        mu = a * x_0 + b * x_t


        if t.min() > 0:
            sigma = torch.sqrt(beta_t)
        else:
            sigma = torch.tensor(0.0).to(x_t.device)

        return mu + sigma * noise

    @torch.no_grad()
    def sampling(self, n_samples, y1, y2, device="cuda"):
        # Initialize noise
        sample = torch.randn(n_samples, self.in_channels, self.latent_size, device=device)

        all_samples = [sample]

        # Assume cond is a tensor of shape [n_samples, num_conditions]
        cond = torch.cat((y1, y2), dim = -1)
        # cond = self.cond_embedder(cond)

        # Reverse diffusion process
        for t in tqdm(reversed(range(self.timesteps))):
            t_tensor = torch.full((n_samples,), t, dtype=torch.int64, device=device)
            #print("t_tensor shape in sampling: ", t_tensor.shape)
            sample = self._reverse_diffusion_x(sample, t_tensor, torch.randn_like(sample, device=device), cond)
            # sample = self.model(sample, t_tensor, cond)
            sample.clamp_(-1, 1)
            all_samples.append(sample)

        all_samples = torch.stack(all_samples, dim=0)
        return sample.clone(), all_samples.clone()