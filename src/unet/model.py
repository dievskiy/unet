from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, filters_in, filters_out, ksize):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=ksize, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(filters_out)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=filters_out, out_channels=filters_out, kernel_size=ksize, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(filters_out)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        conv1_out = self.conv1(x)
        b1_out = self.batch_norm1(conv1_out)
        relu1_out = self.relu1(b1_out)

        conv2_out = self.conv2(relu1_out)
        b2_out = self.batch_norm2(conv2_out)
        relu2_out = self.relu2(b2_out)
        return relu2_out


class EncBlock(nn.Module):
    def __init__(self, filters_in, filters_out, ksize):
        super().__init__()

        self.conv_block = ConvBlock(filters_in=filters_in, filters_out=filters_out, ksize=ksize)
        self.max_pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        enc_out = self.conv_block(x)
        pool_out = self.max_pool(enc_out)
        return enc_out, pool_out


class DecBlock(nn.Module):
    def __init__(self, filters_in, filters_out, ksize):
        super().__init__()
        self.convtr = nn.ConvTranspose2d(
            in_channels=filters_in, out_channels=filters_out, kernel_size=2, stride=2, padding=0
        )
        self.conv_block = ConvBlock(filters_in, filters_out, ksize)

    def forward(self, x, skip):
        enc_out = self.convtr(x)
        with_skipped = torch.cat([enc_out, skip], dim=1)
        out = self.conv_block(with_skipped)
        return out


class UNet(nn.Module):
    def __init__(self, num_channels: int, ladder_size: int = 4):
        super().__init__()

        self.ladder_size = ladder_size
        ksize = (3, 3)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        filters_base = 64

        i = 0

        for _ in range(self.ladder_size):
            if i == 0:
                f_in = num_channels
                f_out = filters_base
            else:
                f_in = filters_base * (2 ** (i - 1))
                f_out = filters_base * (2**i)

            enc_block = EncBlock(filters_in=f_in, filters_out=f_out, ksize=ksize)
            self.encoders.append(enc_block)

            i += 1

        self.enc_mid = ConvBlock(
            filters_in=filters_base * (2 ** (i - 1)), filters_out=filters_base * (2**i), ksize=ksize
        )

        for _ in range(self.ladder_size):
            f_in = filters_base * (2**i)
            f_out = filters_base * (2 ** (i - 1))

            dec_block = DecBlock(filters_in=f_in, filters_out=f_out, ksize=ksize)
            self.decoders.append(dec_block)

            i -= 1

        self.final = nn.Conv2d(filters_base, num_channels, kernel_size=1, padding="same")

    def forward(self, x):
        encs_out = []

        for i, encoder in enumerate(self.encoders):
            if i != 0:
                x = encs_out[i - 1][1]
            encs_out.append(encoder(x))

        mid_out = self.enc_mid(encs_out[len(self.encoders) - 1][1])

        x = mid_out

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, encs_out[len(self.decoders) - i - 1][0])

        final_out = self.final(x)
        return final_out
