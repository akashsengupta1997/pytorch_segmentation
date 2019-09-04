import torch
import torch.nn as nn
from models.utils import Conv2DReLU, Conv2DBatchNormReLU


class UNet2ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True, batch_norm=True):
        super(UNet2ConvBlock, self).__init__()

        if batch_norm:
            self.conv_block1 = Conv2DBatchNormReLU(in_channels, out_channels, 3, 1, 0)
            self.conv_block2 = Conv2DBatchNormReLU(out_channels, out_channels, 3, 1, 0)
        else:
            self.conv_block1 = Conv2DReLU(in_channels, out_channels, 3, 1, 0)
            self.conv_block2 = Conv2DReLU(out_channels, out_channels, 3, 1, 0)

        self.pooling = pooling
        if pooling:
            self.maxpooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv_block1(x)
        outputs = self.conv_block2(x)
        if self.pooling:
            outputs = self.maxpooling(outputs)
        return outputs


class UNetUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='upconv'):
        super(UNetUpConvBlock, self).__init__()

        self.conv_block = UNet2ConvBlock(in_channels, out_channels, pooling=False,
                                         batch_norm=False)

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsampling':
            self. up = nn.UpsamplingBilinear2d(scale_factor=2)

    def centre_crop(self, input, new_height, new_width):
        _, _, height, width = input.size()
        height_diff = (height - new_height) // 2
        width_diff = (width - new_width) // 2

        cropped = input[:, :, height_diff: height_diff + new_height,
                  width_diff: width_diff + new_width]

        print("cropped size", cropped.size())
        return cropped

    def forward(self, x, input_to_concat):
        up = self.up(x)
        cropped = self.centre_crop(input_to_concat, x.shape[2], x.shape[3])
        concat = torch.cat([up, cropped], dim=1)
        outputs = self.conv_block(concat)
        return outputs


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, batch_norm=True, up_mode='upconv'):
        super(UNet, self).__init__()

        # Contraction
        self.contraction1 = UNet2ConvBlock(in_channels, 64, pooling=True, batch_norm=batch_norm)
        self.contraction2 = UNet2ConvBlock(64, 128, pooling=True, batch_norm=batch_norm)
        self.contraction3 = UNet2ConvBlock(128, 256, pooling=True, batch_norm=batch_norm)
        self.contraction4 = UNet2ConvBlock(256, 512, pooling=True, batch_norm=batch_norm)

        self.centre = UNet2ConvBlock(512, 1024, pooling=False, batch_norm=batch_norm)

        # Expansion
        self.expansion4 = UNetUpConvBlock(1024, 512, up_mode=up_mode)
        self.expansion3 = UNetUpConvBlock(512, 256, up_mode=up_mode)
        self.expansion2 = UNetUpConvBlock(256, 128, up_mode=up_mode)
        self.expansion1 = UNetUpConvBlock(128, 64, up_mode=up_mode)

        # Output 1x1 conv
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.contraction1(x)
        x = self.contraction2(x)
        x = self.contraction3(x)
        x = self.contraction4(x)

        centre = self.centre(x)

        x = self.expansion4(centre)
        x = self.expansion3(x)
        x = self.expansion2(x)
        x = self.expansion1(x)

        outputs = self.final(x)
        return outputs



