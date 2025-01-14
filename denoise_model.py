import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

###########
def test_gaussian_properties(z_T, significance_level=0.05):
    """Test if z_T follows N(0,I)"""
    z_np = z_T.cpu().detach().numpy()

    # Test for zero mean
    mean_test = np.mean(z_np, axis=0)
    mean_p_value = stats.ttest_1samp(z_np, 0).pvalue

    # Test for unit variance
    var_test = np.var(z_np, axis=0)
    var_p_value = stats.chi2.sf((z_np.shape[0] - 1) * var_test, df=z_np.shape[0] - 1)

    # Mardia's test for multivariate normality
    def mardia_test(data):
        n = data.shape[0]
        p = data.shape[1]
        data_centered = data - np.mean(data, axis=0)
        S = np.cov(data_centered.T)
        S_inv = np.linalg.inv(S)

        # Skewness
        b1p = np.mean([np.sum((data_centered @ S_inv @ data_centered[j].T) ** 3)
                       for j in range(n)])
        skew_stat = (n / 6) * b1p
        skew_df = p * (p + 1) * (p + 2) / 6
        skew_p_value = stats.chi2.sf(skew_stat, skew_df)

        # Kurtosis
        b2p = np.mean([np.sum((data_centered @ S_inv @ data_centered[j].T) ** 2)
                       for j in range(n)])
        kurt_stat = (b2p - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
        kurt_p_value = 2 * (1 - stats.norm.cdf(abs(kurt_stat)))

        return skew_p_value, kurt_p_value

    mardia_skew_p, mardia_kurt_p = mardia_test(z_np)

    # Calculate overall confidence score
    confidence_score = np.mean([
        np.mean(mean_p_value),
        np.mean(var_p_value),
        mardia_skew_p,
        mardia_kurt_p
    ])

    is_gaussian = all([
        np.all(mean_p_value > significance_level),
        np.all(var_p_value > significance_level),
        mardia_skew_p > significance_level,
        mardia_kurt_p > significance_level
    ])

    return {
        'is_gaussian': is_gaussian,
        'confidence_score': confidence_score,
        'details': {
            'mean_p_value': mean_p_value,
            'var_p_value': var_p_value,
            'mardia_skew_p': mardia_skew_p,
            'mardia_kurt_p': mardia_kurt_p
        }
    }
###########



# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    x_noisy_input = x_noisy.unsqueeze(1)
    predicted_noise = denoise_model(x_noisy_input, t, cond).squeeze(1)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Denoise model
class DenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond):
        super(DenoiseNN, self).__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_layers = [nn.Linear(input_dim+d_cond, hidden_dim)] + [nn.Linear(hidden_dim+d_cond, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim) for i in range(n_layers-1)]
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, t, cond):
        cond = torch.reshape(cond, (-1, self.n_cond))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)
        t = self.time_mlp(t)
        for i in range(self.n_layers-1):
            x = torch.cat((x, cond), dim=1)
            x = self.relu(self.mlp[i](x))+t
            x = self.bn[i](x)
        x = self.mlp[self.n_layers-1](x)
        return x


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    x_input = x.unsqueeze(1)
    predicted_noise = model(x_input, t, cond)  # Will be [batch_size, 1, latent_dim]
    predicted_noise = predicted_noise.squeeze(1)  # Convert back to [batch_size, latent_dim] for the rest of the calculations
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs



@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads=4, d_head=64):
        super().__init__()
        self.scale = d_head ** -0.5
        self.n_heads = n_heads
        inner_dim = d_head * n_heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: t.view(*t.shape[:-1], self.n_heads, -1), (q, k, v))
        q, k, v = map(lambda t: t.transpose(-2, -3), (q, k, v))
        
        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.transpose(-2, -3).contiguous()
        out = out.view(*out.shape[:-2], -1)
        return self.to_out(out)

class ImprovedDenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond):
        super().__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        
        # Enhanced condition processing
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.LayerNorm(d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
            nn.LayerNorm(d_cond),
            nn.ReLU(),
        )
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Cross attention layers for better conditioning
        self.attention_layers = nn.ModuleList([
            CrossAttention(hidden_dim, d_cond)
            for _ in range(n_layers)
        ])
        
        # Main processing layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'block': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ),
                'time_proj': nn.Linear(hidden_dim, hidden_dim),
                'cond_proj': nn.Linear(d_cond, hidden_dim)
            })
            for _ in range(n_layers)
        ])
        
        self.final_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t, cond):
        # Process condition
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)
        
        # Process time embedding
        t = self.time_mlp(t)
        
        # Initial projection - maintain the middle dimension
        h = self.input_proj(x)  # x is [batch_size, 1, input_dim]
        
        # Main processing with attention and residual connections
        for i in range(self.n_layers):
            residual = h
            
            # Apply main block
            h = self.blocks[i]['block'](h)
            
            # Add time embedding
            time_emb = self.blocks[i]['time_proj'](t)
            h = h + time_emb.unsqueeze(1)
            
            # Apply cross attention with condition
            h = self.attention_layers[i](h, cond.unsqueeze(1))
            
            # Add condition projection
            cond_emb = self.blocks[i]['cond_proj'](cond)
            h = h + cond_emb.unsqueeze(1)
            
            # Add residual
            h = h + residual
        
        # Final projection while maintaining shape [batch_size, 1, input_dim]
        return self.final_proj(h)  # Returns [batch_size, 1, input_dim]