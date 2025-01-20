
import torch
from torch import nn


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale_transform = nn.Linear(style_dim, num_features)
        self.style_shift_transform = nn.Linear(style_dim, num_features)

    def forward(self, x, style):
        normalized = self.norm(x)
        scale = self.style_scale_transform(style).unsqueeze(2).unsqueeze(3).expand_as(normalized)
        shift = self.style_shift_transform(style).unsqueeze(2).unsqueeze(3).expand_as(normalized)
        return scale * normalized + shift


class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, style_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.adain = AdaIN(style_dim, out_features)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, style)
        x = self.lrelu(x)
        return x


class Generator_(nn.Module):
    def __init__(self, style_dim, init_features=512, output_channels=3):
        super().__init__()
        self.initial = nn.Parameter(torch.randn(1, init_features, 4, 4))
        self.upblocks = nn.ModuleList([
            UpBlock(init_features // (2 ** i), init_features // (2 ** (i + 1)), style_dim) for i in range(5)
        ])
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(init_features // 32),  # Adjusted for the final features
            nn.LeakyReLU(0.2),
            nn.Conv2d(init_features // 32, output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, style):
        x = self.initial.expand(style.size(0), -1, -1, -1)
        for upblock in self.upblocks:
            x = upblock(x, style)  # Now this should work without error
        return self.to_rgb(x)

import torch
from torch import nn
from torch.nn import functional as F

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, kernel_size=3, padding=1, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.demodulate = demodulate
        self.scale = 1 / (in_channels * kernel_size ** 2) ** 0.5

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.modulation = nn.Linear(style_dim, in_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        batch_size = x.size(0)
        style = self.modulation(style).view(batch_size, 1, self.in_channels, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)

        weight = weight.view(batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        x = x.view(1, batch_size * self.in_channels, x.shape[2], x.shape[3])
        x = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
        x = x.view(batch_size, self.out_channels, x.shape[2], x.shape[3])
        x = self.leaky_relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, style_dim, init_features=512, output_channels=3):
        super().__init__()
        self.initial = nn.Parameter(torch.randn(1, init_features, 4, 4))
        self.style_dim = style_dim
        self.layers = nn.ModuleList([
            ModulatedConv2d(init_features // (2 ** i), init_features // (2 ** (i+1)), style_dim) for i in range(5)
        ])
        self.to_rgb = nn.Sequential(
            nn.Conv2d(init_features // 32, output_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, style):
        x = self.initial.repeat(style.size(0), 1, 1, 1)
        for layer in self.layers:
            x = F.interpolate(x, scale_factor=2)
            x = layer(x, style)
        x = self.to_rgb(x)
        return x





