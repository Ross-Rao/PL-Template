# python import
# package import
import torch
import torch.nn as nn
# local import

__all__ = ['DDPMModule', 'PeUNet']

class DDPMModule(nn.Module):
    def __init__(self, n_steps: int, min_beta: float, max_beta: float):
        super().__init__()
        self.register_buffer('betas', torch.linspace(min_beta, max_beta, n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        self.n_steps = n_steps

    def sample_forward(self, x, t, eps):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * eps

    def sample_backward(self, shape, net, simple_var):
        x = torch.randn(shape, device=self.betas.device)
        for t in reversed(range(self.n_steps)):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var):
        t_vec = torch.full((x_t.shape[0], 1), t, device=self.betas.device)
        eps_theta = net(x_t, t_vec)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        if simple_var or t == 0:
            var = 1 - alpha / (1 - alpha_bar)
        else:
            var = self.betas[t] * (1 - self.alpha_bars[t - 1]) / (1 - alpha_bar)
        z = torch.randn_like(x_t) if t > 0 else 0
        x_prev = 1 / torch.sqrt(alpha) * (x_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta) + torch.sqrt(var) * z
        return x_prev


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
    def __init__(self, shape, channels, pe_dim, n_steps):
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
            UNetBlock((channels[-1], h_list[-1], w_list[-1]), channels[-1], channels[-1])
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


if __name__ == '__main__':
    n_step = 1000
    net = PeUNet((3, 64, 64), [64, 128, 256], 128, n_step)
    x = torch.randn(4, 3, 64, 64)
    t = torch.randint(0, n_step, (4, 1))
    eps_theta = net(x, t)
    print(eps_theta.shape)

    ddpm = DDPMModule(n_step, 1e-4, 0.02)
    eps = torch.randn_like(x)
    x_t = ddpm.sample_forward(x, t, eps)
    eps_theta = net(x_t, t)
    loss = nn.MSELoss()(eps_theta, eps)
    print(loss.item())

    with torch.no_grad():
        shape = (16, 3, 64, 64)
        imgs = ddpm.sample_backward(shape, net, simple_var=True).detach().cpu()
        print(imgs.shape)
        print(imgs.min(), imgs.max(), imgs.mean())

