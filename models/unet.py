import torch
import torch.nn as nn
from models.utils import Conv2DReLU, Conv2DBatchNormReLU


class UNet2ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(UNet2ConvBlock, self).__init__()

        if batch_norm:
            self.conv_block1 = Conv2DBatchNormReLU(in_channels, out_channels, 3, 1, 0)
            self.conv_block2 = Conv2DBatchNormReLU(out_channels, out_channels, 3, 1, 0)
        else:
            self.conv_block1 = Conv2DReLU(in_channels, out_channels, 3, 1, 0)
            self.conv_block2 = Conv2DReLU(out_channels, out_channels, 3, 1, 0)

    def forward(self, x):
        x = self.conv_block1(x)
        outputs = self.conv_block2(x)
        return outputs


class UNetUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='upconv'):
        super(UNetUpConvBlock, self).__init__()

        self.conv_block = UNet2ConvBlock(in_channels, out_channels, batch_norm=False)

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

        return cropped

    def forward(self, x, input_to_concat):
        up = self.up(x)
        cropped = self.centre_crop(input_to_concat, up.shape[2], up.shape[3])
        concat = torch.cat([up, cropped], dim=1)
        outputs = self.conv_block(concat)
        return outputs


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, batch_norm=True, up_mode='upconv'):
        super(UNet, self).__init__()

        # Contraction
        self.conv_block1 = UNet2ConvBlock(in_channels, 64, batch_norm=batch_norm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block2 = UNet2ConvBlock(64, 128, batch_norm=batch_norm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_block3 = UNet2ConvBlock(128, 256, batch_norm=batch_norm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv_block4 = UNet2ConvBlock(256, 512, batch_norm=batch_norm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.centre = UNet2ConvBlock(512, 1024, batch_norm=batch_norm)

        # Expansion
        self.expansion4 = UNetUpConvBlock(1024, 512, up_mode=up_mode)
        self.expansion3 = UNetUpConvBlock(512, 256, up_mode=up_mode)
        self.expansion2 = UNetUpConvBlock(256, 128, up_mode=up_mode)
        self.expansion1 = UNetUpConvBlock(128, 64, up_mode=up_mode)

        # Output 1x1 conv
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        conv_block1 = self.conv_block1(x)
        contract1 = self.maxpool1(conv_block1)
        conv_block2 = self.conv_block2(contract1)
        contract2 = self.maxpool2(conv_block2)
        conv_block3 = self.conv_block3(contract2)
        contract3 = self.maxpool3(conv_block3)
        conv_block4 = self.conv_block4(contract3)
        contract4 = self.maxpool3(conv_block4)

        centre = self.centre(contract4)

        expand4 = self.expansion4(centre, conv_block4)
        expand3 = self.expansion3(expand4, conv_block3)
        expand2 = self.expansion2(expand3, conv_block2)
        expand1 = self.expansion1(expand2, conv_block1)

        outputs = self.final(expand1)
        return outputs



