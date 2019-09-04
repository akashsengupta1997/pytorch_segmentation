import torch.nn as nn


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
