from torch.nn import MaxPool2d

from andraz.models.slimmable_ops import SlimmableConv2d, SwitchableBatchNorm2d
import torch
from torch import nn


class SlimSqueezeUNet(nn.Module):
    def __init__(self, out_channels):
        super(SlimSqueezeUNet, self).__init__()
        self.conv1 = self.input_conv([3, 3, 3, 3], [8, 16, 32, 64], 3, 2)

        self.conv2 = FireModule([8, 16, 32, 64], [4, 8, 16, 32], [16, 32, 64, 128], 3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = FireModule(
            [16, 32, 64, 128], [8, 16, 32, 64], [32, 64, 128, 256], 3
        )
        self.conv4 = FireModule(
            [32, 64, 128, 256], [8, 16, 32, 64], [64, 128, 256, 512], 3
        )

        self.upconv2_ex = self.expand_block([64, 128, 256, 512], [16, 32, 64, 128], 3)
        self.upconv2 = FireModule(
            [16, 32, 64, 128], [8, 16, 32, 64], [32, 64, 128, 256], 3
        )
        self.upconv1 = SlimmableConv2d(
            [16, 32, 48, 64],
            [out_channels, out_channels, out_channels, out_channels],
            3,
            padding="same",
        )

        # self.conv1 = self.contract_block([3, 3, 3, 3], [16, 32, 48, 64], 3)
        # self.conv2 = self.contract_block([16, 32, 48, 64], [32, 64, 96, 128], 3)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.upconv2_ex = self.expand_block([32, 64, 96, 128], [16, 32, 48, 64], 3)
        # self.upconv2 = self.contract_block([32, 64, 96, 128], [16, 32, 48, 64], 3)
        # self.upconv1 = SlimmableConv2d(
        #     [16, 32, 48, 64],
        #     [out_channels, out_channels, out_channels, out_channels],
        #     3,
        #     padding="same",
        # )

        self.layers = [
            self.conv1,
            self.conv2,
            self.maxpool,
            self.upconv2_ex,
            self.upconv2,
            self.upconv1,
        ]

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        y2 = self.upconv2_ex(x2)
        y2 = torch.cat((x1, y2), dim=1)
        y2 = self.upconv2(y2)

        y1 = self.upconv1(y2)
        return y1

    def set_width(self, width):
        for layer in self.layers:
            try:
                for x in layer:
                    if (
                        x.__class__.__name__ == "SlimmableConv2d"
                        or x.__class__.__name__ == "SwitchableBatchNorm2d"
                    ):
                        x.width_mult = width
            except TypeError:
                layer.width_mult = width

    def input_conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            SlimmableConv2d(in_channels, out_channels, kernel_size, padding="same"),
            MaxPool2d(kernel_size=kernel_size, stride=stride),
        )

    def expand_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            SlimmableConv2d(in_channels, out_channels, kernel_size, padding="same"),
            SwitchableBatchNorm2d(out_channels),
            # nn.ReLU(),
        )

    # def contract_block(self, in_channels, out_channels, kernel_size):
    #     return nn.Sequential(
    #         SlimmableConv2d(in_channels, out_channels, kernel_size, padding="same"),
    #         SwitchableBatchNorm2d(out_channels),
    #         nn.ReLU(),
    #         SlimmableConv2d(out_channels, out_channels, kernel_size, padding="same"),
    #         SwitchableBatchNorm2d(out_channels),
    #         nn.ReLU(),
    #     )
    #
    # def expand_block(self, in_channels, out_channels, kernel_size):
    #     return nn.Sequential(
    #         nn.Upsample(scale_factor=2),
    #         SlimmableConv2d(in_channels, out_channels, kernel_size, padding="same"),
    #         SwitchableBatchNorm2d(out_channels),
    #         nn.ReLU(),
    #     )


class FireModule:
    def __init__(self, in_channels, squeeze_channels, expand_channels, kernel_size):
        super(FireModule, self).__init__()
        self.conv1 = SlimmableConv2d(
            in_channels, squeeze_channels, kernel_size, padding="same"
        )
        self.bn1 = x = SwitchableBatchNorm2d(squeeze_channels)
        self.conv_left = SlimmableConv2d(
            squeeze_channels, expand_channels, 1, padding="same"
        )
        self.conv_right = SlimmableConv2d(
            squeeze_channels, expand_channels, kernel_size, padding="same"
        )
        self.layers = [self.conv1, self.bn1, self.conv_left, self.conv_right]

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        left = self.conv_left(X)
        right = self.conv_right(X)
        X = torch.cat((left, right), 1)
        return X
