import torch.nn as nn
import torchvision
from models.utils import Conv2DBatchNormReLU


class SegNet2EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet2EncoderBlock, self).__init__()
        self.conv_block1 = Conv2DBatchNormReLU(in_channels, out_channels, 3, 1, 1)
        self.conv_block2 = Conv2DBatchNormReLU(out_channels, out_channels, 3, 1, 1)
        self.maxpooling_with_indices = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        outputs, pooling_indices = self.maxpooling_with_indices(x)
        return outputs, pooling_indices


class SegNet3EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet3EncoderBlock, self).__init__()

        self.conv_block1 = Conv2DBatchNormReLU(in_channels, out_channels, 3, 1, 1)
        self.conv_block2 = Conv2DBatchNormReLU(out_channels, out_channels, 3, 1, 1)
        self.conv_block3 = Conv2DBatchNormReLU(out_channels, out_channels, 3, 1, 1)
        self.maxpooling_with_indices = nn.MaxPool2d(2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        outputs, pooling_indices = self.maxpooling_with_indices(x)
        return outputs, pooling_indices


class SegNet2DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet2DecoderBlock, self).__init__()

        self.max_unpooling = nn.MaxUnpool2d(2, stride=2)
        self.conv_block1 = Conv2DBatchNormReLU(in_channels, out_channels, 3, 1, 1)
        self.conv_block2 = Conv2DBatchNormReLU(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, indices):
        x = self.max_unpooling(x, indices=indices)
        x = self.conv_block1(x)
        outputs = self.conv_block2(x)
        return outputs


class SegNet3DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet3DecoderBlock, self).__init__()

        self.max_unpooling = nn.MaxUnpool2d(2, stride=2)
        self.conv_block1 = Conv2DBatchNormReLU(in_channels, out_channels, 3, 1, 1)
        self.conv_block2 = Conv2DBatchNormReLU(out_channels, out_channels, 3, 1, 1)
        self.conv_block3 = Conv2DBatchNormReLU(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, indices):
        x = self.max_unpooling(x, indices=indices)
        x = self.conv_block1(x)
        outputs = self.conv_block2(x)
        return outputs


class SegNet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(SegNet, self).__init__()

        self.encoder1 = SegNet2EncoderBlock(in_channels, 64)
        self.encoder2 = SegNet2EncoderBlock(64, 128)
        self.encoder3 = SegNet3EncoderBlock(128, 256)
        self.encoder4 = SegNet3EncoderBlock(256, 512)
        self.encoder5 = SegNet3EncoderBlock(512, 512)
        self.init_vgg16_params()

        self.decoder5 = SegNet3DecoderBlock(512, 512)
        self.decoder4 = SegNet3DecoderBlock(512, 256)
        self.decoder3 = SegNet3DecoderBlock(256, 128)
        self.decoder2 = SegNet3DecoderBlock(128, 64)
        self.decoder1 = SegNet3DecoderBlock(64, num_classes)

    def forward(self, x):
        enc1_out, enc1_indices = self.encoder1(x)
        enc2_out, enc2_indices = self.encoder2(enc1_out)
        enc3_out, enc3_indices = self.encoder3(enc2_out)
        enc4_out, enc4_indices = self.encoder4(enc3_out)
        enc5_out, enc5_indices = self.encoder5(enc4_out)

        dec5_out = self.decoder5(enc5_out, enc5_indices)
        dec4_out = self.decoder4(dec5_out, enc4_indices)
        dec3_out = self.decoder3(dec4_out, enc3_indices)
        dec2_out = self.decoder2(dec3_out, enc2_indices)
        dec1_out = self.decoder1(dec2_out, enc1_indices)

        return dec1_out

    def init_vgg16_params(self):
        """
        SegNet encoder uses same convolutional architecture as VGG16 classification network
        (without the fully-connected classification head). Thus, we can use pre-trained
        VGG16 weights to speed up training. This function initialises SegNet encoder network
        using VGG16 weights.
        """
        encoder_parts = [self.encoder1, self.encoder2, self.encoder3, self.encoder4,
                         self.encoder5]

        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16_features = list(vgg16.features.children())  # Accessing child modules
        vgg16_conv_layers = []
        for layer in vgg16_features:
            if isinstance(layer, nn.Conv2d):
                vgg16_conv_layers.append(layer)

        segnet_conv_layers = []
        for index, part in enumerate(encoder_parts):
            if index < 2:
                conv2d_bn_relu_blocks = [part.conv_block1.conv2d_bn_relu,
                                         part.conv_block2.conv2d_bn_relu]
            else:
                conv2d_bn_relu_blocks = [part.conv_block1.conv2d_bn_relu,
                                         part.conv_block2.conv2d_bn_relu,
                                         part.conv_block3.conv2d_bn_relu]
            for block in conv2d_bn_relu_blocks:
                for layer in block:
                    if isinstance(layer, nn.Conv2d):
                        segnet_conv_layers.append(layer)

        assert(len(segnet_conv_layers) == len(vgg16_conv_layers))

        for vgg_l, segnet_l in zip(vgg16_conv_layers, segnet_conv_layers):
            assert(vgg_l.weight.shape == segnet_l.weight.shape)
            assert(vgg_l.bias.shape == segnet_l.bias.shape)
            segnet_l.weight.data = vgg_l.weight.data
            segnet_l.bias.data = vgg_l.bias.data

        print('VGG16 ImageNet weights initialised!')






