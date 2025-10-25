# python import
# package import
import torch
import torch.nn as nn


def linear_betas(n_steps, min_beta, max_beta):
    return torch.linspace(min_beta, max_beta, n_steps)

def cosine_betas(n_steps, s, max_beta):
    steps = n_steps + 1
    x = torch.linspace(0, n_steps, steps)
    alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, max=max_beta)

# list(range(0, num_steps // ddim_steps, c))


class GaussDiffusion(nn.Module):
    def __init__(self, betas, posterior_log_variance_min):
        super().__init__()
        self.register_buffer('betas', betas)
        self.num_timesteps = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.tensor([0.0])], dim=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        p_lv_min = posterior_log_variance_min if posterior_log_variance_min is not None else posterior_variance[1]
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=p_lv_min)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))


if __name__ == "__main__":
    # Example usage
    n_steps = 1000
    betas_linear = linear_betas(n_steps, 0.0001, 0.02)
    betas_cosine = cosine_betas(n_steps, s=0.008, max_beta=0.999)

    diffusion_linear = GaussDiffusion(betas_linear, posterior_log_variance_min=None)
    diffusion_cosine = GaussDiffusion(betas_cosine, posterior_log_variance_min=None)