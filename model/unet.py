import os
import copy
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from attn import SelfAttention, CrossAttention


class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = torch.nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = torch.nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.nn.functional.silu(x)
        x = self.linear_2(x)
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = torch.nn.GroupNorm(32, in_channels)
        self.conv_feature = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = torch.nn.Linear(n_time, out_channels)

        self.groupnorm_merged = torch.nn.GroupNorm(32, out_channels)
        self.conv_merged = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = torch.nn.Identity()
        else:
            self.residual_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = torch.nn.functional.silu(feature)
        feature = self.conv_feature(feature)

        time = torch.nn.functional.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = torch.nn.functional.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = torch.nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = torch.nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = torch.nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = torch.nn.LayerNorm(channels)
        self.linear_geglu_1 = torch.nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = torch.nn.Linear(4 * channels, channels)

        self.conv_output = torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * torch.nn.functional.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


class Upsample(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(torch.nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = torch.nn.ModuleList([
            SwitchSequential(torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(torch.nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            SwitchSequential(torch.nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(640, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
            SwitchSequential(torch.nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
        ])
        self.bottleneck = SwitchSequential(
            ResidualBlock(1280, 1280),
            AttentionBlock(8, 160),
            ResidualBlock(1280, 1280),
        )
        self.decoders = torch.nn.ModuleList([
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class FinalLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(32, in_channels)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = torch.nn.functional.silu(x)
        x = self.conv(x)
        return x


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.label_embedding = torch.nn.Embedding(500, 768)
        self.unet = UNet()
        self.final = FinalLayer(320, 4)

    def forward(self, latent, context, time):
        time = self.time_embedding(time)
        context = self.label_embedding(context)

        output = self.unet(latent, context, time)
        output = self.final(output)
        return output


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model: torch.nn.Module, current_model: torch.nn.Module):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    @staticmethod
    def reset_parameters(ema_model: torch.nn.Module, model: torch.nn.Module):
        ema_model.load_state_dict(model.state_dict())


class Trainer:
    def __init__(self,
                 noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 epochs=100,
                 learning_rate=5e-3,
                 train_dataloader=None,
                 val_dataloader=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.epochs = epochs
        self.beta = self.prepare_noise_schedule().cuda()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.model = Diffusion().cuda()
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=learning_rate,
                                                             steps_per_epoch=10, epochs=epochs)
        self.mse = torch.nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def get_num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_time_steps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x: torch.Tensor, t: torch.Tensor):
        # Add noise to images at step t
        eps = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, 4, 64, 64)).cuda()
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().cuda()
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    predicted_noise = torch.lerp(model(x, t, None), predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        # rescale image
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.

        if train:
            self.model.train()
            p_bar = self.train_dataloader
        else:
            self.model.eval()
            p_bar = self.val_dataloader

        for i, (images, labels) in enumerate(p_bar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.cuda()
                labels = labels.cuda()
                t = self.sample_time_steps(images.shape[0]).cuda()
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                p_bar.comment = f"train_mse: {loss.item():.2e}, learning_rate: {self.scheduler.get_last_lr()[0]:.2e}"

        avg_loss /= len(p_bar)
        return avg_loss

    def log_images(self):
        labels = torch.arange(np.random.randint(0, 500, size=(8,))).long().cuda()
        sampled_images = self.sample(use_ema=False, labels=labels)
        sample_grid = torchvision.utils.make_grid(sampled_images, nrow=4)

        plt.imshow(sample_grid.cpu().permute(1, 2, 0).numpy(), vmin=0., vmax=1.)
        plt.axis('off')
        plt.show()

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        ema_sample_grid = torchvision.utils.make_grid(ema_sampled_images, nrow=4)

        plt.imshow(ema_sample_grid.cpu().permute(1, 2, 0).numpy(), vmin=0., vmax=1.)
        plt.axis('off')
        plt.show()

    def save_model(self, save_path=None):
        torch.save(self.model.state_dict(), os.path.join(save_path, "diffusion.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(save_path, "ema_diffusion.pt"))

    def load_model(self, save_path=None):
        self.model.load_state_dict(torch.load(os.path.join(save_path, "diffusion.pt")))
        self.ema_model.load_state_dict(torch.load(os.path.join(save_path, "ema_diffusion.pt")))

    def fit(self, log_every_epoch=10, do_validation=False):
        for epoch in range(self.epochs):
            loss = self.one_epoch(train=True)
            print(f"Epoch {epoch + 1}/{self.epochs}:", loss)

            # validation
            if do_validation and self.val_dataloader is not None:
                avg_loss = self.one_epoch(train=False)
                print("val_mse:", avg_loss)

            if (epoch + 1) % log_every_epoch == 0:
                self.log_images()  # log predictions
                self.save_model()  # save model
