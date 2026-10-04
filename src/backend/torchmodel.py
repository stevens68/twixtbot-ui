import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=40, bn_eps=1e-3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=bn_eps)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=bn_eps)

    def forward(self, x):
        h = torch.abs(self.bn1(self.conv1(x)))
        h = torch.abs(self.bn2(self.conv2(h)))
        return h + x


class TwixtNet(nn.Module):
    """PyTorch equivalent of twixtbot's mkbig.py network."""

    def __init__(
        self,
        use_recents=False,
        channels=40,
        blocks=12,
        value_hidden=80,
        value_triple=False,
        value_reductions=2,
        value_padding="VALID",
        bn_eps=1e-3,
    ):
        super().__init__()
        loc_channels = 3 if use_recents else 2
        self.use_recents = use_recents
        self.value_triple = value_triple

        self.location = nn.Conv2d(loc_channels, channels, 1, bias=False)
        self.pegs = nn.Conv2d(2, channels, 5, padding=2, bias=False)
        self.links = nn.Conv2d(8, channels, 4, bias=False)
        self.primary_bn = nn.BatchNorm2d(channels, eps=bn_eps)

        self.blocks = nn.ModuleList(
            [ResidualBlock(channels, bn_eps=bn_eps) for _ in range(blocks)]
        )

        self.policy_conv1 = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2, eps=bn_eps)
        self.policy_conv2 = nn.Conv2d(2, 1, 1, bias=False)

        self.value_padding = value_padding.upper()
        value_padding = 0
        self.value_conv = nn.ModuleList([
            nn.Conv2d(channels, channels, 5, stride=2, padding=value_padding, bias=False)
            for _ in range(value_reductions)
        ])
        self.value_bn = nn.ModuleList([
            nn.BatchNorm2d(channels, eps=bn_eps)
            for _ in range(value_reductions)
        ])

        spatial = 24
        for _ in range(value_reductions):
            if value_padding:
                spatial = (spatial + 1) // 2
            else:
                spatial = (spatial - 5) // 2 + 1
        self.value_fc = nn.Linear(channels * spatial * spatial, value_hidden, bias=False)
        self.value_bn_fc = nn.BatchNorm1d(value_hidden, eps=bn_eps)
        self.value_out = nn.Linear(
            value_hidden, 3 if value_triple else 1, bias=False
        )

    def forward(self, pegs, links, locs):
        # TensorFlow inputs are NHWC; PyTorch uses NCHW.
        pegs = pegs.permute(0, 3, 1, 2)
        links = links.permute(0, 3, 1, 2)
        links = F.pad(links, (1, 2, 1, 2))
        locs = locs.permute(0, 3, 1, 2)

        h = self.location(locs) + self.pegs(pegs) + self.links(links)
        h = torch.abs(self.primary_bn(h))

        for block in self.blocks:
            h = block(h)

        policy = torch.abs(self.policy_bn(self.policy_conv1(h)))
        policy = self.policy_conv2(policy)
        policy = policy[:, :, 1:-1, :].reshape(policy.shape[0], -1)

        v = h
        for conv, bn in zip(self.value_conv, self.value_bn):
            if self.value_padding == "SAME":
                height, width = v.shape[-2:]
                out_h = (height + 1) // 2
                out_w = (width + 1) // 2
                pad_h = max((out_h - 1) * 2 + 5 - height, 0)
                pad_w = max((out_w - 1) * 2 + 5 - width, 0)
                v = F.pad(
                    v,
                    (
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    ),
                )
            v = torch.abs(bn(conv(v)))
        v = v.flatten(1)
        v = torch.abs(self.value_bn_fc(self.value_fc(v)))
        value = self.value_out(v)
        if not self.value_triple:
            value = torch.tanh(value)
        return value, policy
