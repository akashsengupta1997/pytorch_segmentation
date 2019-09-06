import torch.nn as nn
from models.utils import Conv2DBatchNormReLU, ResidualBlock
import numpy as np
import torch


class PSPNet(nn.Module):
    def __init__(self, num_classes, resnet_type='50', in_channels=3):
        super(PSPNet, self).__init__()

        assert resnet_type in ['50', '101']
        if resnet_type == 50:
            num_units_list = [3, 4, 6, 3]
        else:
            num_units_list = [3, 4, 23, 3]

        # Encoder
        self.convbnrelu1_1 = Conv2DBatchNormReLU(in_channels, 64, kernel_size=3, padding=1,
                                                 stride=2, bias=False)
        self.convbnrelu1_2 = Conv2DBatchNormReLU(64, 64, kernel_size=3, padding=1, stride=1,
                                                 bias=False)
        self.convbnrelu1_3 = Conv2DBatchNormReLU(64, 128, kernel_size=3, padding=1, stride=1,
                                                 bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # Vanilla residual blocks
        self.res_block1 = ResidualBlock(num_units_list[0], 128, 64, 256, 1, dilation=1)
        self.res_block2 = ResidualBlock(num_units_list[1], 256, 128, 512, 2, dilation=1)

        # Dilated residual blocks
        self.res_block3 = ResidualBlock(num_units_list[2], 512, 256, 1024, 1, dilation=2)
        self.res_block4 = ResidualBlock(num_units_list[3], 1024, 512, 2048, 1, dilation=4)

        # Pyramid pooling module

        # Final conv layers

        # Aux layers for training

        # Aux loss function

    def forward(self, x):
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)

        x = self.maxpool1(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        return x

x = np.random.randn(4, 3, 256, 256).astype(np.float32)
x = torch.from_numpy(x)
test = PSPNet(10)

out = test(x)
