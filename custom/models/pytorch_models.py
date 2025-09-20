# python import
# package import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
# local import


class ResidualAdd(nn.Module):
    def __init__(self, fn, downsample):
        super().__init__()
        self.fn = fn
        self.downsample = downsample

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += self.downsample(res)
        return x


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x) * self.stddev
        return x


class ResBlockDown(nn.Sequential):
    def __init__(self, in_ch, out_ch, stddev):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    GaussianNoise(stddev=stddev),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    GaussianNoise(stddev=stddev),
                ),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0),
            )
        )


class ResBlockUp(nn.Sequential):
    def __init__(self, in_ch, out_ch, stddev):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    GaussianNoise(stddev=stddev),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    GaussianNoise(stddev=stddev),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
                )
            )
        )


class Encoder(nn.Sequential):
    def __init__(self, img_ch, latent_dim, stddev):
        super().__init__(
            nn.Conv2d(img_ch, 64, kernel_size=3, padding=1),
            ResBlockDown(64, 128, stddev=stddev),
            ResBlockDown(128, 256, stddev=stddev),
            ResBlockDown(256, 512, stddev=stddev),
            ResBlockDown(512, 512, stddev=stddev),
            nn.Flatten(),
            GaussianNoise(stddev=stddev),
            nn.Linear(512 * 4 * 4, latent_dim),
            GaussianNoise(stddev=stddev),
            nn.BatchNorm1d(latent_dim),
        )


class Generator(nn.Sequential):
    def __init__(self, out_ch, latent_dim, stddev):
        super().__init__(
            nn.Linear(latent_dim, 512 * 4 * 4),
            GaussianNoise(stddev=stddev),
            nn.Unflatten(1, (512, 4, 4)),
            ResBlockUp(512, 512, stddev=stddev),
            ResBlockUp(512, 512, stddev=stddev),
            ResBlockUp(512, 256, stddev=stddev),
            ResBlockUp(256, 128, stddev=stddev),
            nn.Conv2d(128, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )


class GaussNoiseDiscriminator(nn.Sequential):
    def __init__(self, latent_dim, hidden_dim, leaky_relu_slope, dropout_rate):
        super().__init__(
            nn.Linear(latent_dim, hidden_dim), nn.BatchNorm1d(hidden_dim, affine=False), nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim, affine=False), nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim, affine=False), nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim, affine=False), nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )


class AlteredLatentFeatRecognizer(nn.Sequential):
    def __init__(self, in_ch, latent_dim, leaky_relu_slope):
        super().__init__(
            nn.Conv2d(in_ch, 32, 5, 2, padding=2), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(32, 64, 5, 2, padding=2), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(64, 128, 5, 2, padding=2), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(128, 256, 5, 2, padding=2), nn.LeakyReLU(leaky_relu_slope),
            nn.Flatten(),                          # 256*4*4 = 4096
            nn.Linear(4096, 512), nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(512, latent_dim)
        )


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        # ----------- 编码器 ----------- #
        self.conv1i = nn.Conv2d(in_ch,  64, 3, padding=1)   # keep 64
        self.conv1  = nn.Conv2d(64,    64, 3, stride=2, padding=1)  # 32

        self.conv2i = nn.Conv2d(64,   128, 3, padding=1)   # keep 32
        self.conv2  = nn.Conv2d(128,  128, 3, stride=2, padding=1)  # 16

        self.conv3i = nn.Conv2d(128,  256, 3, padding=1)   # keep 16
        self.conv3  = nn.Conv2d(256,  256, 3, stride=2, padding=1)  # 8

        self.conv4i = nn.Conv2d(256,  512, 3, padding=1)   # keep 8
        self.conv4  = nn.Conv2d(512,  512, 3, stride=2, padding=1)  # 4

        # ----------- 瓶颈 ----------- #
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)    # 4

        # ----------- 解码器 ----------- #
        # 上采样：nearest + 1×1 conv 调整通道（与 Keras 一致）
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4->8
            nn.Conv2d(1024, 512, 1)          # 1×1 降通道
        )
        self.conv6 = nn.Conv2d(512*2, 512, 3, padding=1)

        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8->16
            nn.Conv2d(512, 256, 1)
        )
        self.conv7 = nn.Conv2d(256*2, 256, 3, padding=1)

        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16->32
            nn.Conv2d(256, 128, 1)
        )
        self.conv8 = nn.Conv2d(128*2, 128, 3, padding=1)

        self.up9 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32->64
            nn.Conv2d(128, 64, 1)
        )
        self.conv9 = nn.Conv2d(64*2, 64, 3, padding=1)

        # ----------- 输出 ----------- #
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        def he_normal(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(he_normal)

    def forward(self, x):
        # 编码
        c1i = F.relu(self.conv1i(x))
        c1  = F.relu(self.conv1(c1i))

        c2i = F.relu(self.conv2i(c1))
        c2  = F.relu(self.conv2(c2i))

        c3i = F.relu(self.conv3i(c2))
        c3  = F.relu(self.conv3(c3i))

        c4i = F.relu(self.conv4i(c3))
        c4  = F.relu(self.conv4(c4i))

        c5 = F.relu(self.conv5(c4))

        # 解码 + 跳跃连接
        u6 = self.up6(c5)
        u6 = torch.cat([c4i, u6], dim=1)
        c6 = F.relu(self.conv6(u6))

        u7 = self.up7(c6)
        u7 = torch.cat([c3i, u7], dim=1)
        c7 = F.relu(self.conv7(u7))

        u8 = self.up8(c7)
        u8 = torch.cat([c2i, u8], dim=1)
        c8 = F.relu(self.conv8(u8))

        u9 = self.up9(c8)
        u9 = torch.cat([c1i, u9], dim=1)
        c9 = F.relu(self.conv9(u9))

        out = torch.sigmoid(self.conv10(c9))
        return out


class VGG19Binary(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features          # 冻结或微调随你
        self.avgpool  = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)   # (B, 2)


# if __name__ == "__main__":
#     # Test single models
#     img = torch.randn(2, 1, 64, 64).cuda()
#     encoder = Encoder(img_ch=1, latent_dim=350, stddev=0.03).cuda()
#     gen = Generator(out_ch=1, latent_dim=350, stddev=0.03).cuda()
#     disc = GaussNoiseDiscriminator(latent_dim=350, hidden_dim=2048, leaky_relu_slope=0.2, dropout_rate=0.5).cuda()
#     recognizer = AlteredLatentFeatRecognizer(in_ch=1, latent_dim=128, leaky_relu_slope=0.2).cuda()
#     unet = UNet(in_ch=1, out_ch=1).cuda()
#     cls = VGG19Binary(num_classes=2, dropout=0.5).cuda()
#
#     z = encoder(img)
#     print("Encoded shape:", z.shape)
#     recon_img = gen(z)
#     print("Reconstructed image shape:", recon_img.shape)
#     disc_out = disc(z)
#     print("Discriminator output shape:", disc_out.shape)
#     recog_out = recognizer(recon_img)
#     print("Recognizer output shape:", recog_out.shape)
#     unet_out = unet(img)
#     print("UNet output shape:", unet_out.shape)
#
#     from torchinfo import summary
#     summary(encoder, input_size=(2, 1, 64, 64))
#     summary(gen, input_size=(2, 350))
#     summary(disc, input_size=(2, 350))
#     summary(recognizer, input_size=(2, 1, 64, 64))
#     summary(unet, input_size=(2, 1, 64, 64))


class ExplainModel(nn.Module):
    def __init__(self, img_ch, num_classes, latent_dim, disc_hidden_dim, subset_dim, stddev, leaky_relu_slope, dropout_rate):
        super().__init__()
        self.encoder = Encoder(img_ch, latent_dim, stddev)
        self.generator = Generator(img_ch, latent_dim, stddev)
        self.unet = UNet(img_ch, img_ch)
        self.recognizer = AlteredLatentFeatRecognizer(img_ch, latent_dim, leaky_relu_slope)
        self.disc = GaussNoiseDiscriminator(latent_dim, disc_hidden_dim, leaky_relu_slope, dropout_rate)
        self.neu = nn.Sequential(nn.Linear(subset_dim, num_classes))

        self.subset_dim = subset_dim

    def forward(self, x, pert_vec):
        z = self.encoder(x)
        z_subset_score = self.neu(z[:, :self.subset_dim])

        recon_x = self.generator(z)
        super_x = self.unet(x)

        disc_z = self.disc(z)
        pert_x = self.generator(z + pert_vec)
        altered_z = self.recognizer(pert_x - recon_x)

        return recon_x, super_x, z_subset_score, z, disc_z, altered_z


# if __name__ == "__main__":
#     # Test the integrated model
#     img = torch.randn(2, 1, 64, 64).cuda()
#     pert = torch.randn(2, 350).cuda() * 0.1
#     model = ExplainModel(
#         img_ch=1, num_classes=2, latent_dim=350, disc_hidden_dim=2048, subset_dim=14,
#         stddev=0.03, leaky_relu_slope=0.2, dropout_rate=0.5
#     ).cuda()
#     out = model(img, pert)
#     for k in out:
#         print(k.shape)
#     from torchinfo import summary
#     summary(model, input_size=[(2, 1, 64, 64), (2, 350)])


def cov_loss_terms(z_batch: torch.Tensor, k, eps):
    """
    z_batch: (B, D)
    返回: cov, z_std_loss, diag_cov_mean, off_diag_loss, off_diag_k_mean
    """
    z = z_batch - z_batch.mean(dim=0)
    z_var = z.var(dim=0) + eps          # 方差
    z_std_loss = F.relu(1 - z_var.sqrt()).mean()

    cov = (z.T @ z) / z.size(0)         # (D, D)
    diag_cov = torch.diag(cov).abs().mean()

    off_diag = cov.clone()
    off_diag.fill_diagonal_(0)
    off_diag_loss = off_diag.abs().mean()

    off_diag_k = off_diag[:k, :k]
    off_diag_k_mean = off_diag_k.abs().mean()

    return cov, z_std_loss, diag_cov, off_diag_loss, off_diag_k_mean


class CovLoss(nn.Module):
    def __init__(self, k, eps):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, z_batch):
        _, z_std_loss, diag_cov, _, _ = cov_loss_terms(z_batch, self.k, self.eps)
        return 0.5 * z_std_loss + 0.5 * diag_cov


class PerceptualVGGLoss(nn.Module):
    def __init__(self, ch, vgg, layers, std, mean):
        super().__init__()
        self.vgg_model = vgg
        self.layers = layers
        self.register_buffer('std', torch.tensor(std).view(1,ch,1,1))
        self.register_buffer('mean', torch.tensor(mean).view(1,ch,1,1))

    def cal_features(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for name, module in self.vgg_model.named_children():  # name 是 str 数字
            x = module(x)
            if name in self.layers:
                feats.append(x)
        return feats

    def forward(self, raw_img, recon_img):
        # raw_img, recon_img = raw_img.repeat(1, 3, 1, 1), recon_img.repeat(1, 3, 1, 1)
        pred_feats = self.cal_features(raw_img)
        true_feats = self.cal_features(recon_img)
        losses = [abs(t - p).mean() for t, p in zip(true_feats, pred_feats)]
        return torch.stack(losses).mean()


def z_mean_var_loss(y_pred):
    return torch.mean(torch.abs(torch.var(y_pred, dim=0)-1) + torch.abs(torch.mean(y_pred)))


def js_loss(logits1, logits2, temperature=1.0, reduction='batchmean'):
    """
    Jensen-Shannon 散度损失
    logits1, logits2: [B, C]  未归一化 logits
    reduction: 同 F.kl_div 的选项
    返回: JS(p||q)  值域 [0, ln2]，越小越接近
    """
    p = F.softmax(logits1 / temperature, dim=1)
    q = F.softmax(logits2 / temperature, dim=1)
    m = 0.5 * (p + q)                          # 平均分布

    # KL(p||m) 与 KL(q||m)
    kl_pm = F.kl_div(F.log_softmax(logits1 / temperature, dim=1), m, reduction=reduction)
    kl_qm = F.kl_div(F.log_softmax(logits2 / temperature, dim=1), m, reduction=reduction)

    js = 0.5 * (kl_pm + kl_qm) * (temperature ** 2)
    return js


class TotalLoss(nn.Module):
    def __init__(self, ch, num_classes, clf_path, vgg_layers, cov_k, cov_eps, dropout, loss_weights):
        super().__init__()
        vgg = VGG19Binary(dropout=dropout, num_classes=num_classes).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        ckpt = torch.load(clf_path)
        state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        self.clf = VGG19Binary(dropout=dropout, num_classes=num_classes)
        self.clf.load_state_dict(state_dict)
        self.clf.eval()
        for p in self.clf.parameters():
            p.requires_grad = False

        self.imagenet_perceptual_loss = PerceptualVGGLoss(ch, vgg, vgg_layers, [0.229,0.224,0.225], [0.485,0.456,0.406])
        self.clf_perceptual_loss = PerceptualVGGLoss(ch, self.clf.features, vgg_layers, [1, 1, 1], [0, 0, 0])
        self.cov_loss = CovLoss(cov_k, cov_eps)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.register_buffer('loss_weights', torch.tensor(loss_weights))

    def forward(self, recon_img, super_img, z_subset_score, z, disc_z, altered_z, raw_img, gt_altered):
        loss_dt = {
            'vgg_recon': self.imagenet_perceptual_loss(raw_img, recon_img),
            'clf_recon': self.clf_perceptual_loss(raw_img, recon_img),
            'vgg_super': self.imagenet_perceptual_loss(super_img, raw_img),
            'clf_super': self.clf_perceptual_loss(super_img, raw_img),
            'cov': self.cov_loss(z),
            'z_mean_var': z_mean_var_loss(z),
            'adv': self.bce(disc_z, torch.ones_like(disc_z)),
            'clf_score': js_loss(z_subset_score, self.clf(raw_img)),
            'alter': self.ce(altered_z, gt_altered.float())
        }
        total_loss = sum(w * loss_dt[k] for w, k in zip(self.loss_weights, loss_dt.keys()))
        loss_dt['loss'] = total_loss
        return loss_dt


if __name__ == '__main__':
    # Test the integrated model and loss
    img = torch.randn(2, 3, 64, 64).cuda()
    pert = torch.randn(2, 350).cuda()
    model = ExplainModel(
        img_ch=3, num_classes=2, latent_dim=350, disc_hidden_dim=2048, subset_dim=14,
        stddev=0.03, leaky_relu_slope=0.2, dropout_rate=0.5
    ).cuda()
    loss_fn = TotalLoss(clf_path='/media/lai/cbe9a31e-b62c-4de2-b86d-3d6ea08c8013/Ross/exp_results/DISCOVER/2025-09-20_14-51-49_job/lightning_logs/version_0/checkpoints/last.ckpt',
        ch=3, num_classes=2, vgg_layers=('3', '8', '13', '18', '23', '28', '33'),
        cov_k=10, cov_eps=1e-4, dropout=0.5,
        loss_weights=[1, 1, 1, 1, 1, 1, 0.1, 1, 1]
    ).cuda()

    out = model(img, pert)
    for item in out:
        print(f"{item.shape}")
    loss_dt = loss_fn(*out, img, torch.zeros(2,).cuda())
    for k, v in loss_dt.items():
        print(f"{k}: {v.item()}")
    from torchinfo import summary
    summary(model, input_size=[(2, 3, 64, 64), (2, 350)])
    summary(loss_fn, input_size=[(2, 3, 64, 64), (2, 3, 64, 64), (2, 2), (2, 350), (2, 2), (2, 350), (2, 3, 64, 64), (2,)])
