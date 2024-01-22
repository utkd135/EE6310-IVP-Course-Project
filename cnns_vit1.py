import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torchinfo


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size

        self.attn = nn.MultiheadAttention(embed_dim=inp, num_heads=heads, dropout=dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)
        print(inp)
        self.attn_norm = nn.LayerNorm(inp)
        self.ff_norm = nn.LayerNorm(oup)

        self.attn_proj = nn.Linear(inp, oup)
        self.ff_proj = nn.Linear(oup, oup)

    def forward(self, x):
        # self-attn
        residual = x
        print(x.shape, 'before attn_norm')
        x = self.attn_norm(x)
        print(x.shape, 'attn_norm')
        x = x.view(self.iw*self.ih, x.shape[0], -1) # reshape input to 3D tensor
        x, _ = self.attn(x, x, x)
        x = x.view(self.iw, self.ih, -1)
        x = residual + self.attn_proj(x)

        # feedforward
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + self.ff_proj(x)

        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, depth):
        super().__init__()

        #     CoAtNet((64, 64), 3, num_blocks=[2, 2, 3, 5], channels=[64, 96, 192, 384])

        ih, iw = image_size
        block = {'T': Transformer}

        self.CNNVIT = self._make_layer(block['T'], in_channels, in_channels, depth, (ih, iw))

    def forward(self, x):
        print(x.shape, 'before')
        x = self.CNNVIT(x)
        print(x.shape, 'after')
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            layers.append(block(inp, oup, image_size))
        return nn.Sequential(*layers)


def coatnet_0():
    num_blocks = [2, 2, 3, 5]  # , 2]            # L
    channels = [64, 96, 192, 384]  # , 768]      # D
    return CoAtNet((64, 64), 128, 2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 128, 64, 64)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))
    print(torchinfo.summary(net, (1, 128, 64, 64)))
    print(net)