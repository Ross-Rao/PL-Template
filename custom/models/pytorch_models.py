# python import
# package import
import torch
import torch.nn as nn
# local import

__all__ = ['DDIM', 'PeUNet']

def linear_betas(n_steps, min_beta, max_beta):
    return torch.linspace(min_beta, max_beta, n_steps)

def cosine_betas(n_steps, s, max_beta):
    steps = n_steps + 1
    x = torch.linspace(0, n_steps, steps)
    alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, max=max_beta)

class DDIM(nn.Module):
    def __init__(self, betas, ddim_steps: int, simple_var=True, theta=1.0):
        super(DDIM, self).__init__()
        betas = eval(betas[0])(**betas[1])
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        self.n_steps = betas.shape[0]
        self.ddim_steps = ddim_steps
        self.simple_var = simple_var
        self.theta = theta


    def sample_forward(self, x_0, t, eps):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps

    def sample_backward(self, img: torch.Tensor, net):
        device = img.device
        ts = torch.linspace(self.n_steps, 0, (self.ddim_steps + 1), device=device, dtype=torch.long)
        for i in range(1, self.ddim_steps + 1):
            cur_t, prev_t = ts[i - 1] - 1, ts[i] - 1

            ab_cur = self.alpha_bars[cur_t]
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            t_tensor = torch.full((img.shape[0], 1), cur_t.item(), device=device, dtype=torch.long)
            eps = net(img, t_tensor)

            var = self.theta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(img)
            first_term = (ab_prev / ab_cur).sqrt() * img
            second_term = ((1 - ab_prev - var).sqrt() - (ab_prev * (1 - ab_cur) / ab_cur).sqrt()) * eps
            if self.simple_var:
                third_term = ((1 - ab_cur / ab_prev).sqrt()) * noise
            else:
                third_term = var.sqrt() * noise
            img = first_term + second_term + third_term

        return img


class Residual(nn.Module):
    def __init__(self, fn, downsample):
        super().__init__()
        self.fn = fn
        self.downsample = downsample

    def forward(self, x):
        return self.fn(x) + self.downsample(x)


class ResidualBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            Residual(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                ) if in_channels != out_channels else nn.Identity()
            )
        )

class ResidualAttnBlock(nn.Module):
    """Channel-first self-attention, 残差连接"""
    def __init__(self, ch, num_groups):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, ch)          # 先归一化
        self.qkv   = nn.Conv2d(ch, ch * 3, 1)   # 1×1 生成 Q,K,V
        self.proj  = nn.Conv2d(ch, ch, 1)       # 1×1 输出投影
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)   # (B,C,H,W)
        # 把空间拉成一维，做 scaled dot-product
        q = q.view(B, C, -1).permute(0, 2, 1)   # (B,HW,C)
        k = k.view(B, C, -1)                    # (B,C,HW)
        v = v.view(B, C, -1).permute(0, 2, 1)   # (B,HW,C)
        attn = torch.softmax(q @ k * self.scale, dim=-1)  # (B,HW,HW)
        out = (attn @ v).permute(0, 2, 1).view(B, C, H, W)  # 还原空间
        return x + self.proj(out)               # 残差

class UNetBlock(nn.Sequential):
    def __init__(self, shape, in_channels, out_channels):
        super().__init__(
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.LayerNorm(shape),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    ),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity(),
                ),
                nn.ReLU(inplace=True),
            )
        )


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq, indexing='ij')
        pe_2i = torch.sin(pos / 10000 ** (two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000 ** (two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe

    def forward(self, t):
        self.embedding.requires_grad_(False)
        return self.embedding(t)


class PeUNet(nn.Module):
    def __init__(self, shape, channels, pe_dim, n_steps, attention_groups):
        super().__init__()
        h_list = [int(shape[1] * (0.5 ** i)) for i in range(len(channels))]
        w_list = [int(shape[2] * (0.5 ** i)) for i in range(len(channels))]
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linear_en = nn.ModuleList()
        self.pe_linear_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        prev_channel = shape[0]
        for channel, h, w in zip(channels[:-1], h_list[:-1], w_list[:-1]):
            self.encoders.append(nn.Sequential(
                UNetBlock((prev_channel, h, w), prev_channel, channel),
                UNetBlock((channel, h, w), channel, channel)
            ))
            self.pe_linear_en.append(nn.Sequential(
                nn.Linear(pe_dim, prev_channel), nn.ReLU(inplace=True), nn.Linear(prev_channel, prev_channel)
            ))
            self.downs.append(nn.Conv2d(channel, channel, kernel_size=2, stride=2))
            prev_channel = channel

        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        self.mid = nn.Sequential(
            UNetBlock((prev_channel, h_list[-1], w_list[-1]), prev_channel, channels[-1]),
            UNetBlock((channels[-1], h_list[-1], w_list[-1]), channels[-1], channels[-1]),
            ResidualAttnBlock(channels[-1], attention_groups)
        )

        prev_channel = channels[-1]
        for channel, h, w in zip(reversed(channels[:-1]), reversed(h_list[:-1]), reversed(w_list[:-1])):
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, kernel_size=2, stride=2))
            self.decoders.append(nn.Sequential(
                UNetBlock((channel * 2, h, w), channel * 2, channel),
                UNetBlock((channel, h, w), channel, channel)
            ))
            self.pe_linear_de.append(nn.Linear(pe_dim, prev_channel))
            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, shape[0], 3, 1 ,1)
        self.pe = PositionalEncoding(n_steps, pe_dim)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)  # [n, 1] -> [n, 1, pe_dim]
        skips = []
        for encoder, pe_linear, down in zip(self.encoders, self.pe_linear_en, self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)  # [n, 1, pe_dim] -> [n, 1, c] -> [n, c, 1, 1]
            x = encoder(x + pe)
            skips.append(x)
            x = down(x)
        pe = self.pe_mid(t).reshape(n, -1, 1, 1)  # [n, 1, pe_dim] -> [n, 1, channel[-2]] -> [n, channel[-2], 1, 1]
        x = self.mid(x + pe)
        for decoder, pe_linear, up, encoder_out in zip(self.decoders, self.pe_linear_de, self.ups, reversed(skips)):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = up(x)
            pad_x = (encoder_out.shape[2] - x.shape[2]) // 2
            pad_y = (encoder_out.shape[3] - x.shape[3]) // 2
            x = nn.functional.pad(x, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)
            x = decoder(torch.cat((encoder_out, x), dim=1) + pe)
        x = self.conv_out(x)
        return x

if __name__ == "__main__":
    # Example usage
    n_steps = 1000
    ddim_steps = 50
    min_beta = 0.0001
    max_beta = 0.02

    betas = linear_betas(n_steps, min_beta, max_beta)
    ddim_model = DDIM(betas, ddim_steps)

    # Dummy data and model for testing
    x_0 = torch.randn(4, 3, 64, 64)  # Batch of 4 images
    t = torch.randint(0, n_steps, (4,))
    eps = torch.randn_like(x_0)

    # Forward sampling
    x_t = ddim_model.sample_forward(x_0, t, eps)

    # Dummy neural network model
    class DummyNet(nn.Module):
        def forward(self, x, t):
            return torch.randn_like(x)

    net = DummyNet()

    # Backward sampling
    sampled_img = ddim_model.sample_backward(x_t, net)
    print("Sampled image shape:", sampled_img.shape)