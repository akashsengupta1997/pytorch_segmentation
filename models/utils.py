import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Conv2DReLU(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size, stride, padding, bias=True,
                 dilation=1):
        super(Conv2DReLU, self).__init__()

        conv2d = nn.Conv2d(in_channels, n_filters, kernel_size, stride=stride, padding=padding,
                           bias=bias, dilation=dilation)

        self.conv2d_relu = nn.Sequential(conv2d,
                                         nn.ReLU())

    def forward(self, x):
        outputs = self.conv2d_relu(x)
        return outputs


class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size, stride, padding, bias=True,
                 dilation=1):
        super(Conv2DBatchNorm, self).__init__()

        conv2d = nn.Conv2d(in_channels, n_filters, kernel_size, stride=stride, padding=padding,
                           bias=bias, dilation=dilation)

        self.conv2d_bn = nn.Sequential(conv2d,
                                         nn.BatchNorm2d(n_filters))

    def forward(self, x):
        outputs = self.conv2d_bn(x)
        return outputs


class Conv2DBatchNormReLU(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size, stride, padding, bias=True,
                 dilation=1):
        super(Conv2DBatchNormReLU, self).__init__()

        conv2d = nn.Conv2d(in_channels, n_filters, kernel_size, stride=stride, padding=padding,
                           bias=bias, dilation=dilation)

        self.conv2d_bn_relu = nn.Sequential(conv2d,
                                            nn.BatchNorm2d(n_filters),
                                            nn.ReLU())

    def forward(self, x):
        outputs = self.conv2d_bn_relu(x)
        return outputs


class ConvBottleNeckUnit(nn.Module):
    """
    Bottle-beck unit with projection shortcut (for matching channel dimension size) from
    ResNet architecture. Can use dilated convolutions here (to increase receptive field) -
    as in PSPNet architecture.
    """
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation=1):
        super(ConvBottleNeckUnit, self).__init__()

        self.conv2d_bn_relu1 = Conv2DBatchNormReLU(in_channels, mid_channels, 1, 1, 0,
                                                   bias=False)

        # if dilation > 1:
        self.conv2d_bn_relu2 = Conv2DBatchNormReLU(mid_channels, mid_channels, 3, stride,
                                                   padding=dilation, bias=False,  # padding cancels out dilation's change of height/width
                                                   dilation=dilation)
        # else:
        #     self.conv2d_bn_relu2 = Conv2DBatchNormReLU(mid_channels, mid_channels, 3, stride,
        #                                                padding=1, bias=False,
        #                                                dilation=dilation)

        self.conv2d_bn3 = Conv2DBatchNorm(mid_channels, out_channels, 1, 1, 0, bias=False)

        # Projection shortcut changes input channel dimension to out_channels.
        self.conv2d_bnshortcut = Conv2DBatchNorm(in_channels, out_channels, 1, stride, 0,
                                                 bias=False)

        self.final_relu = nn.ReLU()

    def forward(self, x):
        outputs = self.conv2d_bn3(self.conv2d_bn_relu2(self.conv2d_bn_relu1(x)))
        residuals = self.conv2d_bnshortcut(x)
        outputs = self.final_relu(outputs + residuals)
        return outputs


class IdentityBottleNeckUnit(nn.Module):
    """
       Bottle-beck unit with identity shortcut (for matching channel dimension size) from
       ResNet architecture. Can use dilated convolutions here (to increase receptive field) -
       as in PSPNet architecture.
       """

    def __init__(self, in_channels, mid_channels, dilation=1):
        super(IdentityBottleNeckUnit, self).__init__()

        self.conv2d_bn_relu1 = Conv2DBatchNormReLU(in_channels, mid_channels, 1, 1, 0,
                                                   bias=False)

        # if dilation > 1:
        self.conv2d_bn_relu2 = Conv2DBatchNormReLU(mid_channels, mid_channels, 3, 1,
                                                   padding=dilation, bias=False,  # padding cancels out dilation's change of height/width
                                                   dilation=dilation)
        # else:
        #     self.conv2d_bn_relu2 = Conv2DBatchNormReLU(mid_channels, mid_channels, 3, 1,
        #                                                padding=1, bias=False,
        #                                                dilation=dilation)

        self.conv2d_bn3 = Conv2DBatchNorm(mid_channels, in_channels, 1, 1, 0, bias=False)

        self.final_relu = nn.ReLU()

    def forward(self, x):
        outputs = self.conv2d_bn3(self.conv2d_bn_relu2(self.conv2d_bn_relu1(x)))
        outputs = self.final_relu(outputs + x)

        return outputs


class ResidualBlock(nn.Module):
    """
    Series of bottlebeck units, starting with projection shortcut followed by several identity
    shortcuts. Can use dilated convolutions, as in PSPNet architecture.
    """
    def __init__(self, num_units, in_channels, mid_channels, out_channels, stride,
                 dilation=1):
        super(ResidualBlock, self).__init__()

        units = [ConvBottleNeckUnit(in_channels, mid_channels, out_channels, stride,
                                    dilation)]

        for i in range(num_units - 1):
            units.append(IdentityBottleNeckUnit(out_channels, mid_channels, dilation))

        self.res_block = nn.Sequential(*units)

    def forward(self, x):
        return self.res_block(x)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()

        out_channels = int(in_channels / len(pool_sizes))
        self.convs = []
        for i in range(len(pool_sizes)):
            self.convs.append(Conv2DBatchNormReLU(in_channels, out_channels, 1, 1, 0))

        self.pool_sizes = pool_sizes

    def forward(self, x):
        height, width = x.shape[2:]

        kernel_sizes = []
        strides = []

        for pool_size in self.pool_sizes:
            kernel_sizes.append((int(height/pool_size), int(width/pool_size)))
            strides.append((int(height/pool_size), int(width/pool_size)))

        outputs_to_concat = [x]
        for i, (conv, pool_size) in enumerate(zip(self.convs, self.pool_sizes)):
            pooled = F.avg_pool2d(x, kernel_sizes[i], stride=strides[i], padding=0)
            conved = conv(pooled)
            out = F.interpolate(conved, size=(height, width), mode='bilinear',
                                align_corners=True)
            outputs_to_concat.append(out)

        return torch.cat(outputs_to_concat, dim=1)


x = torch.from_numpy(np.random.randn(4, 2048, 32, 32).astype(np.float32))





