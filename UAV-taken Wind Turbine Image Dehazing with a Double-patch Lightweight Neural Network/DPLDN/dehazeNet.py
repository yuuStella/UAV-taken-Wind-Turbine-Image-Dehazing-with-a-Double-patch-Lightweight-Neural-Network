# @author: hayat
import torch
import torch.nn as nn
import math
import torch


class LightDehaze_Net(nn.Module):

    def __init__(self):
        super(LightDehaze_Net, self).__init__()

        # DehazeNet Architecture
        self.relu = nn.ReLU(inplace=True)
        self.depth_conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=3
        )
        self.point_conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.depth_conv2 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=3
        )
        self.point_conv2 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.depth_conv3 = nn.Conv2d(
            in_channels=6,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=6
        )
        self.point_conv3 = nn.Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.depth_conv4 = nn.Conv2d(
            in_channels=6,
            out_channels=6,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=6
        )
        self.point_conv4 = nn.Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.depth_conv5 = nn.Conv2d(
            in_channels=15,
            out_channels=15,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=15
        )
        self.point_conv5 = nn.Conv2d(
            in_channels=15,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

        self.e_conv_layer6 = nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.e_conv_layer7 = nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=2, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.e_conv_layer8 = nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=3, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

    def forward(self, img):
        pipeline = []
        pipeline.append(img)

        out = self.depth_conv1(img)
        conv_layer1 = self.relu(self.point_conv1(out))
        conv_layer6 = self.relu(self.e_conv_layer6(conv_layer1))
        conv_layer7 = self.relu(self.e_conv_layer7(conv_layer1))
        conv_layer8 = self.relu(self.e_conv_layer8(conv_layer1))

        concat_layer1 = torch.cat((conv_layer1, conv_layer7), 1)

        out = self.depth_conv3(concat_layer1)
        conv_layer3 = self.relu(self.point_conv3(out))

        concat_layer2 = torch.cat((conv_layer7, conv_layer3), 1)

        out = self.depth_conv4(concat_layer2)
        conv_layer4 = self.relu(self.point_conv4(out))

        concat_layer3 = torch.cat((conv_layer6, conv_layer8, conv_layer1, conv_layer3, conv_layer4), 1)

        out = self.depth_conv5(concat_layer3)
        conv_layer5 = self.relu(self.point_conv5(out))

        dehaze_image = self.relu((conv_layer5 * img) - conv_layer5 + 1)
        # J(x) = clean_image, k(x) = x8, I(x) = x, b = 1

        return dehaze_image














