# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
import torch
from torch import nn
import torch.nn.functional as F


class ResED(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=3,
        depth=3,
        wf=6,
        padding=2,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(ResED, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            next_channels = 2 ** (wf + i)
            self.down_path.append(
                ResED_DownBlock(prev_channels, next_channels, padding, batch_norm)
            )
            prev_channels = next_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(1, depth)):
            next_channels = 2 ** (wf + i - 1)
            self.up_path.append(
                ResED_UpBlock(prev_channels, next_channels, up_mode, padding, batch_norm)
            )
            prev_channels = next_channels
            
        self.up_path.append(
            ResED_UpBlock(prev_channels, n_classes, up_mode, padding, batch_norm)
        )
        self.last_out_channel = n_classes
    def forward(self, x):
        skip_connections = []
        for i, down in enumerate(self.down_path):
            x, skip = down(x)
            skip_connections.append(skip)

        for i, up in enumerate(self.up_path):
            x = up(x, skip_connections[-i - 1])
            
        # flatten the pixels with thier right depth
        # print(x.shape)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x


class ResED_DownBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(ResED_DownBlock, self).__init__()
        block1 = []
        block1.append(nn.Conv2d(in_size, out_size, kernel_size=(3, 3), stride=padding, padding=padding))
        block2 = []
        block2.append(nn.ReLU())
        if batch_norm:
            block2.append(nn.BatchNorm2d(out_size))

        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)

    def forward(self, x):
        skip = self.block1(x)
        out = self.block2(skip)
        # print("down: out.shape:", out.shape)
        # print("down: skip.shape:", skip.shape)

        return out, skip


class ResED_UpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(ResED_UpBlock, self).__init__()
        if up_mode == 'upconv':
            stride = padding
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=(3, 3), stride=stride, padding=padding)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

    def forward(self, x, skip):
        # print("up: x.shape:", x.shape)
        # print("up: skip.shape:", skip.shape)
        out = self.up(x + skip) # skip connection
        return out
