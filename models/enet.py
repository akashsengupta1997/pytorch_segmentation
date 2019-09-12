import torch
import torch.nn as nn
from models.utils import Conv2DBatchNormPReLU, ConvTransposed2DBatchNormPReLU
import numpy as np


class InitialBlock(nn.Module):
    def __init__(self, num_filters, in_channels=3):
        super(InitialBlock, self).__init__()

        self.conv_branch = Conv2DBatchNormPReLU(in_channels, num_filters, 3, 2, 1, bias=False)
        self.pool_branch = nn.MaxPool2d(2)

    def forward(self, x):
        conv = self.conv_branch(x)
        pool = self.pool_branch(x)
        outputs = torch.cat([conv, pool], dim=1)
        return outputs


class BottleneckUnit(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, dilation=1,
                 asymmetric=False, dropout_prob=0.01, projection_ratio=4):
        super(BottleneckUnit, self).__init__()

        mid_channels = int(in_channels / projection_ratio)
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
        outputs = self.final_prelu(shortcut_branch + conv_branch)
        return outputs


class DownsamplingBottleneckUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.01, projection_ratio=4):
        super(DownsamplingBottleneckUnit, self).__init__()

        mid_channels = int(in_channels / projection_ratio)
        self.conv2d_bn_prelu_reduce_channels = Conv2DBatchNormPReLU(in_channels, mid_channels,
                                                                    2, 2, 0, bias=False)
        self.conv2d_bn_prelu_increase_channels = Conv2DBatchNormPReLU(mid_channels,
                                                                      out_channels, 1, 1, 0,
                                                                      bias=False)
        self.conv2d_main = Conv2DBatchNormPReLU(mid_channels,
                                                mid_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
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
        if shortcut_branch.is_cuda:
            padding_zeros = padding_zeros.cuda()
        shortcut_branch = torch.cat([shortcut_branch, padding_zeros], dim=1)

        outputs = self.final_prelu(shortcut_branch + conv_branch)
        return outputs, pool_indices


class UpsamplingBottleneckUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.01, projection_ratio=4):
        super(UpsamplingBottleneckUnit, self).__init__()

        mid_channels = int(out_channels / projection_ratio)
        self.conv2d_bn_prelu_reduce_channels = Conv2DBatchNormPReLU(in_channels, mid_channels,
                                                                    1, 1, 0, bias=False)
        self.conv2d_bn_prelu_increase_channels = Conv2DBatchNormPReLU(mid_channels,
                                                                      out_channels, 1, 1, 0,
                                                                      bias=False)
        self.convtransposed2d_main = ConvTransposed2DBatchNormPReLU(mid_channels,
                                                                    mid_channels,
                                                                    kernel_size=3,
                                                                    stride=2,
                                                                    padding=1,
                                                                    bias=False,
                                                                    output_padding=1)
        self.regulariser = nn.Dropout2d(dropout_prob)

        self.conv2d_shortcut = Conv2DBatchNormPReLU(in_channels, out_channels, 1, 1, 0,
                                                    bias=False)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.final_prelu = nn.PReLU(out_channels)

    def forward(self, x, pooling_indices):
        conv_branch = self.conv2d_bn_prelu_reduce_channels(x)
        conv_branch = self.convtransposed2d_main(conv_branch)
        conv_branch = self.conv2d_bn_prelu_increase_channels(conv_branch)
        conv_branch = self.regulariser(conv_branch)

        shortcut_branch = self.conv2d_shortcut(x)
        shortcut_branch = self.max_unpool(shortcut_branch, pooling_indices)

        outputs = self.final_prelu(shortcut_branch + conv_branch)
        return outputs


class ENet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(ENet, self).__init__()

        self.initial_block = InitialBlock(13)

        # Stage 1
        dropout_prob = 0.01
        self.stage1_down = DownsamplingBottleneckUnit(16, 64, dropout_prob=dropout_prob)
        stage1_layers = []
        for _ in range(4):
            stage1_layers.append(BottleneckUnit(64, 3, 1, 1, dropout_prob=dropout_prob))
        self.stage1_bottlenecks = nn.Sequential(*stage1_layers)

        # Stages 2 and 3
        dropout_prob = 0.1
        self.stage2_down = DownsamplingBottleneckUnit(64, 128, dropout_prob=dropout_prob)
        stage2and3_layers = []
        for _ in range(2):
            stage2and3_layers.append(BottleneckUnit(128, 3, 1, 1, dropout_prob=dropout_prob))
            stage2and3_layers.append(BottleneckUnit(128, 3, 1, 2, dilation=2,
                                                    dropout_prob=dropout_prob))
            stage2and3_layers.append(BottleneckUnit(128, 5, 1, 2, asymmetric=True,
                                                    dropout_prob=dropout_prob))
            stage2and3_layers.append(BottleneckUnit(128, 3, 1, 4, dilation=4,
                                                    dropout_prob=dropout_prob))
            stage2and3_layers.append(BottleneckUnit(128, 3, 1, 1, dropout_prob=dropout_prob))
            stage2and3_layers.append(BottleneckUnit(128, 3, 1, 8, dilation=8,
                                                    dropout_prob=dropout_prob))
            stage2and3_layers.append(BottleneckUnit(128, 5, 1, 2, asymmetric=True,
                                                    dropout_prob=dropout_prob))
            stage2and3_layers.append(BottleneckUnit(128, 3, 1, 16, dilation=16,
                                                    dropout_prob=dropout_prob))
        self.stages2and3_bottlenecks = nn.Sequential(*stage2and3_layers)

        # Stage 4
        self.stage4_up = UpsamplingBottleneckUnit(128, 64, dropout_prob=dropout_prob)
        stage4_layers = []
        for _ in range(2):
            stage4_layers.append(BottleneckUnit(64, 3, 1, 1, dropout_prob=dropout_prob))
        self.stage4_bottlenecks = nn.Sequential(*stage4_layers)

        # Stage 5
        self.stage5_up = UpsamplingBottleneckUnit(64, 16, dropout_prob=dropout_prob)
        self.stage5_bottleneck = BottleneckUnit(16, 3, 1, 1, dropout_prob=dropout_prob)

        self.final_conv_transposed = nn.ConvTranspose2d(16,
                                                        num_classes,
                                                        3,
                                                        stride=2,
                                                        padding=1,
                                                        output_padding=1)

    def forward(self, x):

        x = self.initial_block(x)

        x, pool_indices1 = self.stage1_down(x)
        x = self.stage1_bottlenecks(x)

        x, pool_indices2 = self.stage2_down(x)
        x = self.stages2and3_bottlenecks(x)

        x = self.stage4_up(x, pool_indices2)
        x = self.stage4_bottlenecks(x)

        x = self.stage5_up(x, pool_indices1)
        x = self.stage5_bottleneck(x)

        outputs = self.final_conv_transposed(x)
        return outputs






