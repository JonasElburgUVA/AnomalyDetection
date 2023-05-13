from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_blocks


class PixelSNAIL(nn.Module):
    def __init__(
        self,
        d,
        shape=(64, 64),
        n_channels=64,
        n_block=4,
        n_res_block=2,
        dropout_p=0.1,
        downsample=1,
        non_linearity=F.elu
    ):
        super().__init__()

        self.d = d
        self.non_linearity = non_linearity
        self.ini_conv = nn_blocks.MaskedConv(d, n_channels, kernel_size=7, stride=downsample, mask_type="A")

        height, width = shape

        height //= downsample
        width //= downsample

        self.blocks = nn.ModuleList()

        for i in range(n_block):
            self.blocks.append(
                nn_blocks.PixelBlock(
                    n_channels,
                    n_channels,
                    n_res_block=n_res_block,
                    shape=(height, width),
                    dropout_p=dropout_p,
                    non_linearity=non_linearity
            ))

        self.upsample = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=downsample, stride=downsample)
        self.out = nn_blocks.WNConv2d(n_channels, d, 1)

    def forward(self, input):
        input = F.one_hot(input, self.d).permute(0, 3, 1, 2)

        out = self.ini_conv(input)

        for block in self.blocks:
            out = block(out)

        out = self.upsample(self.non_linearity(out))
        out = self.out(self.non_linearity(out))

        return out

    def loss(self, x, reduction="mean"):
        logits = self.forward(x)
        nll = F.cross_entropy(logits, x, reduction=reduction)

        return OrderedDict(loss=nll)

    def sample(self, n, img_size=(64, 64)):
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long().to(device)

        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    # FIXME: Moving this outside the loop should speed it up
                    logits = self(samples)[:, :, r, c]
                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)

        return samples.cpu().numpy()
