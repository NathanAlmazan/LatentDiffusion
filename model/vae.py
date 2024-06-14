import os
import torch
import torchvision
import matplotlib.pyplot as plt

from attn import SelfAttention
from fastprogress import progress_bar


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        x += residue
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = torch.nn.GroupNorm(32, in_channels)
        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = torch.nn.GroupNorm(32, out_channels)
        self.conv_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = torch.nn.Identity()
        else:
            self.residual_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class Encoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            torch.nn.GroupNorm(32, 512),
            torch.nn.SiLU(),
            torch.nn.Conv2d(512, 8, kernel_size=3, padding=1),
            torch.nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = torch.nn.functional.pad(x, (0, 1, 0, 1))
            x = module(x)

        noise = torch.randn(size=(x.shape[0], 64, 64)).cuda()

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise

        return x


class Decoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(4, 4, kernel_size=1, padding=0),
            torch.nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            torch.nn.GroupNorm(32, 128),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class Trainer:
    def __init__(self, dataloader):
        # initialize dataset
        self.dataloader = dataloader
        # initialize variational autoencoder
        self.encoder = Encoder().float().cuda()
        self.decoder = Decoder().float().cuda()

    @staticmethod
    def loss_fn(recon_x, x, mean, log_var):
        mse = torch.nn.functional.mse_loss(recon_x, x, size_average=False)
        kld = -0.5 * torch.mean(1 + log_var - torch.pow(mean, 2) - torch.exp(log_var))
        loss = 500 * mse + kld

        return loss, mse, kld

    def log_images(self, batch):
        self.encoder.eval()
        self.decoder.eval()
        with torch.autocast("cuda") and torch.inference_mode():
            sample_images = batch.cuda()
            mean, z_log_var, z = self.encoder(sample_images)
            reconstructed = self.decoder(z)
            reconstructed = (reconstructed.clamp(-1, 1) + 1) / 2
            reconstructed = (reconstructed * 255).type(torch.uint8)

        # grid samples
        image_grid = torchvision.utils.make_grid(reconstructed[:8], padding=0).cpu()
        image_grid = image_grid.permute(1, 2, 0).numpy()

        # display a reconstructed images
        plt.figure(figsize=(16, 12))
        plt.imshow(image_grid)
        plt.axis('off')
        plt.show()

    def train(self, epochs, lr=5e-3, eps=1e-5, verbose=True, checkpoint=None, save_path=None):
        # initialize optimizers
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr,
                                      eps=eps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                        steps_per_epoch=len(self.dataloader), epochs=epochs)

        # load checkpoint to continue training (optional)
        if checkpoint:
            self.encoder.load_state_dict(torch.load(os.path.join(checkpoint, "encoder.pt")))
            self.decoder.load_state_dict(torch.load(os.path.join(checkpoint, "decoder.pt")))

        for epoch in range(epochs):
            # set model in train mode
            self.encoder.train()
            self.decoder.train()
            # track model performance
            avg_loss = 0.
            avg_kl_loss = 0.
            avg_reconstruction_loss = 0.
            # iterate over batches
            dataset = progress_bar(self.dataloader, leave=False) if verbose else self.dataloader
            for image, _ in dataset:
                image = image.cuda()  # move image to gpu
                # enable mixed precision calculation
                with torch.autocast("cuda") and torch.enable_grad():
                    # forward pass
                    z_mean, z_log_var, z = self.encoder(image)
                    reconstruction = self.decoder(z)
                    # calculate loss
                    loss, mse, kld = self.loss_fn(reconstruction, image, z_mean, z_log_var)
                # backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                # logger
                avg_loss += loss
                avg_kl_loss += kld
                avg_reconstruction_loss += mse
            # log average loss
            avg_loss /= len(self.dataloader)
            avg_kl_loss /= len(self.dataloader)
            avg_reconstruction_loss /= len(self.dataloader)
            print(
                f"Epoch {epoch + 1}/{epochs} - total_loss: {avg_loss:.2e}, reconstruction_loss: {avg_reconstruction_loss:.2e}, kl_loss: {avg_kl_loss:.2e}")

            if (epoch + 1) % 10 == 0:
                # view reconstructed images
                self.log_images(next(iter(self.dataloader)))
                # save checkpoint
                if save_path:
                    torch.save(self.encoder.state_dict(), os.path.join(save_path, "encoder.pt"))
                    torch.save(self.decoder.state_dict(), os.path.join(save_path, "decoder.pt"))
