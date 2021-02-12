import math
import typing

import torch
from torch import nn
import torch.nn.functional as F

import constant


# MARK: - DenseNet blocks
class Bottleneck (nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        """

        :param in_channels:
        :param growth_rate: `k` in the paper.
        """
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)    # Note that BN is before conv.
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out = torch.cat([out, x], 1)    # Cat on the channel dimension.

        return out


class DenseBlock (nn.Module):
    def __init__(self, in_channels: int, n_blocks: int, growth_rate: int):
        super(DenseBlock, self).__init__()

        bottleneck_blocks = []
        for i in range(n_blocks):
            bottleneck_blocks.append(Bottleneck(in_channels, growth_rate))
            in_channels += growth_rate

        self.bottleneck_blocks_sequence = nn.Sequential(*bottleneck_blocks)

    def forward(self, x: torch.Tensor):
        x = self.bottleneck_blocks_sequence(x)
        return x


class Transition (nn.Module):
    """
    A transition layer between two contiguous dense blocks.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = F.avg_pool2d(x, 2)

        return x


# MARK: - DenseNet
class DenseNet (nn.Module):
    def __init__(self, n_classes: int, n_blocks_list: typing.List[int], growth_rate: int, reduction=0.5):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate

        n_channels = growth_rate * 2
        self.conv = nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=False)

        self.dense_block_1 = DenseBlock(in_channels=n_channels, n_blocks=n_blocks_list[0], growth_rate=growth_rate)
        n_channels += n_blocks_list[0] * growth_rate

        out_channels = int(math.floor(n_channels * reduction))
        self.transition1 = Transition(in_channels=n_channels, out_channels=out_channels)
        n_channels = out_channels

        self.dense_block_2 = DenseBlock(in_channels=n_channels, n_blocks=n_blocks_list[1], growth_rate=growth_rate)
        n_channels += n_blocks_list[1] * growth_rate

        out_channels = int(math.floor(n_channels * reduction))
        self.transition2 = Transition(in_channels=n_channels, out_channels=out_channels)
        n_channels = out_channels

        self.dense_block_3 = DenseBlock(in_channels=n_channels, n_blocks=n_blocks_list[2], growth_rate=growth_rate)
        n_channels += n_blocks_list[2] * growth_rate

        out_channels = int(math.floor(n_channels * reduction))
        self.transition3 = Transition(in_channels=n_channels, out_channels=out_channels)
        n_channels = out_channels

        self.dense_block_4 = DenseBlock(in_channels=n_channels, n_blocks=n_blocks_list[3], growth_rate=growth_rate)
        n_channels += n_blocks_list[3] * growth_rate

        self.bn = nn.BatchNorm2d(n_channels)
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.dense_block_1(x)
        x = self.transition1(x)
        x = self.dense_block_2(x)
        x = self.transition2(x)
        x = self.dense_block_3(x)
        x = self.transition3(x)
        x = self.dense_block_4(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    
# MARK: DenseNet convenience
class DenseNet762 (DenseNet):
    def __init__(self):
        super(DenseNet762, self).__init__(n_classes=constant.N_CLASSES, n_blocks_list=[6, 12, 24, 16], growth_rate=12)
