import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_blocks

from nets_AR import PixelSNAIL
from collections import OrderedDict


class VQVAE(nn.Module):
    def __init__(
        self,
        d,
        n_channels,
        code_size,
        n_block=3,
        n_res_block=4,
        dropout_p=.5,
        reconstruction_loss=F.l1_loss
    ):
        super().__init__()

        self.d = d
        self.code_size = code_size
        self.reconstruction_loss = reconstruction_loss

        if isinstance(n_channels,int):
            n_channels = [n_channels] * (n_block + 1)
        else:
            n_block = len(n_channels) - 1

        down = [
            nn.Conv2d(d, n_channels[0], kernel_size=7),
            nn.BatchNorm2d(n_channels[0])
        ]

        for block in range(n_block):
            for _ in range(n_res_block):
                down.append(nn_blocks.GatedResNet(
                    n_channels[block],
                    3,
                    dropout_p=dropout_p,
                    conv=nn.Conv2d,
                    norm=nn.BatchNorm2d)
                )

            down.extend([
                nn.Conv2d(n_channels[block], n_channels[block + 1], kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(n_channels[block + 1])
            ])

        down.append(nn_blocks.GatedResNet(
            n_channels[-1],
            3,
            dropout_p=dropout_p,
            conv=nn.Conv2d,
            norm=nn.BatchNorm2d
        ))

        self.Q = nn.Sequential(*down)
        self.codebook = nn_blocks.Quantize(code_size, n_channels[-1])

        # Decoder
        up = [
            nn.Conv2d(n_channels[-1], n_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels[-1])
        ]

        for block in range(n_block):
            for _ in range(n_res_block):
                up.append(nn_blocks.GatedResNet(
                    n_channels[-(block + 1)],
                    3,
                    dropout_p=dropout_p,
                    conv=nn.Conv2d,
                    norm=nn.BatchNorm2d
                ))
            
            up.extend([
                nn.ConvTranspose2d(n_channels[-(block + 1)], n_channels[-(block + 2)], kernel_size=6, stride=2, padding=2),
                nn.BatchNorm2d(n_channels[-(block + 2)])
            ])
        
        up.append(nn_blocks.GatedResNet(n_channels[0], 3, dropout_p=dropout_p, conv=nn.Conv2d, norm=nn.BatchNorm2d))

        up.extend([
            nn.ELU(),
            nn.Conv2d(n_channels[0], d, kernel_size=1, padding=0)
        ])

        self.P = nn.Sequential(*up)

    def encode(self, x):
        return self.Q(x)

    def decode(self, latents):
        return self.P(latents)

    def forward(self, x):
        z = self.encode(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decode(e_st)

        diff1 = torch.mean((z - e.detach()) ** 2)
        diff2 = torch.mean((e - z.detach()) ** 2)

        return x_tilde, diff1 + diff2

    def loss(self, x, reduction="mean"):
        x_tilde, diff = self(x)

        recon_loss = self.reconstruction_loss(x_tilde, x, reduction=reduction)

        if reduction == "mean":
            loss = recon_loss + diff

        elif reduction == "none":
            loss = torch.mean(recon_loss) + diff

        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff)


class VQLatentSNAIL(PixelSNAIL):
    def __init__(self, feature_extractor_model, **kwargs):
        super().__init__(d=feature_extractor_model.code_size, **kwargs)

        for p in feature_extractor_model.parameters():
            p.requires_grad = False

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor_model.eval()

    def retrieve_codes(self,x):
        with torch.no_grad():
            self.feature_extractor_model.eval()
            z = self.feature_extractor_model.encode(x)
            _, _, code = self.feature_extractor_model.codebook(z)

        return code

    def forward(self, x):
        # Retrieve codes for images
        code = self.retrieve_codes(x)
        return super(VQLatentSNAIL, self).forward(code)

    def forward_latent(self, code):
        return super(VQLatentSNAIL, self).forward(code)

    def loss(self, x, reduction="mean"):
        # Retrieve codes for images
        code = self.retrieve_codes(x)
        logits = super(VQLatentSNAIL,self).forward(code)
        nll = F.cross_entropy(logits, code, reduction=reduction)

        return OrderedDict(loss=nll)

    def sample(self, n, img_size = (64,64)):
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long().to(device)

        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    # FIXME: Moving this outside the loop should speed it up
                    logits = super(VQLatentSNAIL,self).forward(samples)[:, :, r, c]

                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)

        return samples.cpu().numpy()
