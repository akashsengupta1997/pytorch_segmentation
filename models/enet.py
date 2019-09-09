import torch
import torch.nn as nn
from models.utils import Conv2DBatchNormPReLU


class ENetInitialBlock(nn.Module):
    def __init__(self, num_filters, in_channels=3):
        super(ENetInitialBlock, self).__init__()

        self.conv_branch = Conv2DBatchNormPReLU(in_channels, num_filters, 3, 2, 1, bias=False)
        self.pool_branch = nn.MaxPool2d(2)

    def forward(self, x):
        conv = self.conv_branch(x)
        pool = self.pool_branch(x)
        outputs = torch.cat([conv, pool], dim=1)
        return outputs


class ENetBottleneckUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, stride, padding,
                 dilation=1, asymmetric=False, dropout_prob=0.01):
        super(ENetBottleneckUnit, self).__init__()

        self.conv2d_bn_prelu_reduce_channels = Conv2DBatchNormPReLU(in_channels, mid_channels,
                                                                    1, 1, 0, bias=False)
        self.conv2d_bn_prelu_increase_channels = Conv2DBatchNormPReLU(mid_channels,
                                                                      in_channels, 1, 1, 0,
                                                                      bias=False)
        if asymmetric:
            self.conv2d_1 = Conv2DBatchNormPReLU(mid_channels,
                                                 mid_channels,
                                                 kernel_size=(kernel_size, 1),
                                                 stride=stride,
                                                 padding=(padding, 0),
                                                 dilation=dilation,
                                                 bias=False)
            self.conv2d_2 = Conv2DBatchNormPReLU(mid_channels,
                                                 mid_channels,
                                                 kernel_size=(1, kernel_size),
                                                 stride=stride,
                                                 padding=(0, padding),
                                                 dilation=dilation,
                                                 bias=False)
            self.conv2d_main = nn.Sequential(self.conv2d_1, self.conv2d_2)

        else:
            self.conv2d_main = Conv2DBatchNormPReLU(mid_channels,
                                                    mid_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    bias=False)

        self.regulariser = nn.Dropout2d(dropout_prob)
        self.final_prelu = nn.PReLU(in_channels)

    def forward(self, x):
        conv_branch = self.conv2d_bn_prelu_reduce_channels(x)
        conv_branch = self.conv2d_main(conv_branch)
        conv_branch = self.conv2d_bn_prelu_increase_channels(conv_branch)
        conv_branch = self.regulariser(conv_branch)
        shortcut_branch = x
        outputs = shortcut_branch + conv_branch
        return outputs


class ENetDownsamplingBottleneckUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, dropout_prob=0.01):
        super(ENetDownsamplingBottleneckUnit, self).__init__()

        self.conv2d_bn_prelu_reduce_channels = Conv2DBatchNormPReLU(in_channels, mid_channels,
                                                                    2, 2, 0, bias=False)
        self.conv2d_bn_prelu_increase_channels = Conv2DBatchNormPReLU(mid_channels,
                                                                      out_channels, 1, 1, 0,
                                                                      bias=False)
        self.conv2d_main = Conv2DBatchNormPReLU(mid_channels,
                                                mid_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                dilation=dilation,
                                                bias=False)

        self.regulariser = nn.Dropout2d(dropout_prob)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.final_prelu = nn.PReLU(out_channels)

    def forward(self, x):
        conv_branch = self.conv2d_bn_prelu_reduce_channels(x)
        conv_branch = self.conv2d_main(conv_branch)
        conv_branch = self.conv2d_bn_prelu_increase_channels(conv_branch)
        conv_branch = self.regulariser(conv_branch)

        shortcut_branch, pool_indices = self.maxpool2d(x)
        n, c_conv, h, w = conv_branch.size()
        c_input = x.size()[1]
        padding_zeros = torch.zeros(n, c_conv-c_input, h, w)





        outputs = shortcut_branch + conv_branch
        return outputs








