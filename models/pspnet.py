import torch.nn as nn
from models.utils import Conv2DBatchNormReLU, ResidualBlock, PyramidPooling
import torch.nn.functional as F


class PSPNet(nn.Module):
    def __init__(self, num_classes, resnet_type='50', in_channels=3):
        super(PSPNet, self).__init__()

        assert resnet_type in ['50', '101']
        if resnet_type == 50:
            num_units_list = [3, 4, 6, 3]
        else:
            num_units_list = [3, 4, 23, 3]

        # Encoder
        self.convbnrelu1 = Conv2DBatchNormReLU(in_channels, 64, kernel_size=3, padding=1,
                                               stride=2, bias=False)
        self.convbnrelu2 = Conv2DBatchNormReLU(64, 64, kernel_size=3, padding=1, stride=1,
                                               bias=False)
        self.convbnrelu3 = Conv2DBatchNormReLU(64, 128, kernel_size=3, padding=1, stride=1,
                                               bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # Vanilla residual blocks
        self.res_block1 = ResidualBlock(num_units_list[0], 128, 64, 256, 1, dilation=1)
        self.res_block2 = ResidualBlock(num_units_list[1], 256, 128, 512, 2, dilation=1)

        # Dilated residual blocks
        self.res_block3 = ResidualBlock(num_units_list[2], 512, 256, 1024, 1, dilation=2)
        self.res_block4 = ResidualBlock(num_units_list[3], 1024, 512, 2048, 1, dilation=4)

        # Pyramid pooling module
        self.pyramid_pooling = PyramidPooling(2048, [6, 3, 2, 1])

        # Final conv layers
        self.convbnrelu4 = Conv2DBatchNormReLU(4096, 512, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.class_scores = nn.Conv2d(512, num_classes, 1, 1, 0)

        # Aux layers for training
        self.convbnrelu4_aux = Conv2DBatchNormReLU(1024, 256, 3, 1, 1, bias=False)
        self.class_scores_aux = nn.Conv2d(256, num_classes, 1, 1, 0)

    def forward(self, x):
        input_shape = x.shape[2:]

        x = self.convbnrelu1(x)
        x = self.convbnrelu2(x)
        x = self.convbnrelu3(x)

        x = self.maxpool1(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        if self.training:
            x_aux = self.convbnrelu4_aux(x)
            x_aux = self.dropout(x_aux)
            aux_output = self.class_scores_aux(x_aux)

        x = self.res_block4(x)

        x = self.pyramid_pooling(x)

        x = self.convbnrelu4(x)
        x = self.dropout(x)
        x = self.class_scores(x)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        if self.training:
            return [output, aux_output]
        else:
            return output
