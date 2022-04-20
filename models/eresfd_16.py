"""
EResFD
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""
import os
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from models.sepfpn import FullySepFPN

# Efficient Channel Multiplier inside FPN structure
CM_IN_FPN = 1


class CPM3_BN(nn.Module):
    def __init__(self, in_channels):
        super(CPM3_BN, self).__init__()

        self.in_channels = in_channels

        self.res_branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=(3, 3),
                dilation=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(True),
        )

        self.res_branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels // 2,
                in_channels // 4,
                kernel_size=(3, 3),
                dilation=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels // 4,
                in_channels // 4,
                kernel_size=(3, 3),
                dilation=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.res_branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels // 4,
                in_channels // 4,
                kernel_size=(3, 3),
                dilation=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels // 4,
                in_channels // 4,
                kernel_size=(3, 3),
                dilation=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

    def forward(self, x):

        # residual
        res1 = self.res_branch1(x)
        res2 = self.res_branch2(res1)
        res3 = self.res_branch3(res2)
        out = torch.cat([res1, res2, res3], dim=1)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, depth, stride=1):
        super(ConvBlock, self).__init__()

        intermediate_depth = depth

        if in_channel != depth or stride == 2:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        else:
            self.shortcut_layer = None

        self.res_layer = nn.Sequential(
            nn.Conv2d(in_channel, intermediate_depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(intermediate_depth),
            nn.ReLU(True),
            nn.Conv2d(intermediate_depth, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = x

        x = self.res_layer(x)
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(shortcut)

        x += shortcut

        return x


class DetEncoder(nn.Module):
    def __init__(self, num_features=256):
        super(DetEncoder, self).__init__()
        self.num_features = num_features

        self._generate_network()

    def _generate_network(self):

        self.add_module(
            "b2_" + str(1),
            nn.Sequential(
                ConvBlock(
                    self.num_features, int(self.num_features * CM_IN_FPN), stride=2
                ),
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                ),
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                ),
            ),
        )

        self.add_module(
            "b2_" + str(2),
            nn.Sequential(
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                    stride=2,
                ),
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                ),
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                ),
            ),
        )

        self.add_module(
            "b2_" + str(3),
            nn.Sequential(
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                    stride=2,
                ),
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                ),
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                ),
            ),
        )

        self.add_module(
            "b2_" + str(4),
            nn.Sequential(
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                    stride=2,
                ),
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                ),
            ),
        )
        self.add_module(
            "b2_" + str(5),
            nn.Sequential(
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                    stride=2,
                ),
                ConvBlock(
                    int(self.num_features * CM_IN_FPN),
                    int(self.num_features * CM_IN_FPN),
                ),
            ),
        )

        self.add_module(
            "fpn",
            FullySepFPN(
                int(self.num_features * CM_IN_FPN),
                int(self.num_features * CM_IN_FPN),
                4,
                use_bn=True,
                use_relu=True,
            ),
        )
        self.add_module("FEM_0", CPM3_BN(int(self.num_features * CM_IN_FPN)))
        self.add_module("FEM_1", CPM3_BN(int(self.num_features * CM_IN_FPN)))
        self.add_module("FEM_2", CPM3_BN(int(self.num_features * CM_IN_FPN)))
        self.add_module("FEM_3", CPM3_BN(int(self.num_features * CM_IN_FPN)))
        self.add_module("FEM_4", CPM3_BN(int(self.num_features * CM_IN_FPN)))
        self.add_module("FEM_5", CPM3_BN(int(self.num_features * CM_IN_FPN)))

    def forward(self, inp):
        # Feature Pyramid Networks for Object Detection, 2017
        # Hourglass networks for human pose estimation, 2016
        # Cascaded Pyramid Network for Multi-Person Pose Estimation, 2018
        # How far are we from solving the 2D & 3D Face Alignment problem??, 2017
        # FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction, 2019

        conv0 = inp

        conv1 = self._modules["b2_" + str(1)](conv0)

        conv2 = self._modules["b2_" + str(2)](conv1)

        conv3 = self._modules["b2_" + str(3)](conv2)

        conv4 = self._modules["b2_" + str(4)](conv3)

        conv5 = self._modules["b2_" + str(5)](conv4)

        # FPN
        conv0, conv1, conv2, conv3, conv4, conv5 = self._modules["fpn"](
            [conv0, conv1, conv2, conv3, conv4, conv5]
        )

        # FEM forward
        conv0 = self._modules["FEM_0"](conv0)
        conv1 = self._modules["FEM_1"](conv1)
        conv2 = self._modules["FEM_2"](conv2)
        conv3 = self._modules["FEM_3"](conv3)
        conv4 = self._modules["FEM_4"](conv4)
        conv5 = self._modules["FEM_5"](conv5)

        return [conv0, conv1, conv2, conv3, conv4, conv5]


class EResNet(nn.Module):
    def __init__(self, width_mult=1.0):
        super(EResNet, self).__init__()

        # Base part
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, int(128 * width_mult), kernel_size=5, stride=4, padding=2, bias=False
            ),
            nn.BatchNorm2d(int(128 * width_mult)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                int(128 * width_mult),
                int(128 * width_mult),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(128 * width_mult)),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                int(128 * width_mult),
                int(256 * width_mult),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(256 * width_mult)),
            nn.ReLU(True),
        )

        self.conv4 = ConvBlock(int(256 * width_mult), int(256 * width_mult))

        # Stacking part
        self.add_module("m0", DetEncoder(int(256 * width_mult)))

        self._initialize_weights()

        # Auxilary modules conv weight init
        for i in range(3):
            self._auxmodules_initialize_weights(self._modules["m0"]._modules["fpn"])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.PReLU):
                negative_slope = 0.25
                gain = math.sqrt(2.0 / (1.0 + negative_slope ** 2))
                fan = 1
                std = gain / math.sqrt(fan)
                m.weight.data.normal_(0, std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _auxmodules_initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight.data, mean=0.0, std=0.005)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        out_hg = self._modules["m0"](x)
        outputs.append(out_hg)

        return outputs


def _get_base_layers(width_mult=1.0):
    return EResNet(width_mult)


def _get_head_layers(
    width_mult=1.0,
    num_classes=2,
    maxout_bg_size=3,
    anchor_sizes=None,
    anchor_size_ratio=None,
):
    act_loc_layers = []
    act_conf_layers = []
    loc_layers = []
    conf_layers = []

    act_loc_layers += [nn.ReLU(False)]
    act_conf_layers += [nn.ReLU(False)]

    num_anchor_per_pixel = 1
    if anchor_sizes is not None:
        num_anchor_per_pixel *= len(anchor_sizes[0])
    if anchor_size_ratio is not None:
        num_anchor_per_pixel *= len(anchor_size_ratio[0])

    loc_layers += [
        nn.Conv2d(
            int(256 * width_mult * CM_IN_FPN),
            4 * num_anchor_per_pixel,
            kernel_size=1,
            padding=0,
        )
    ]
    conf_layers += [
        nn.Conv2d(
            int(256 * width_mult * CM_IN_FPN),
            (maxout_bg_size + num_classes - 1) * num_anchor_per_pixel,
            kernel_size=1,
            padding=0,
        )
    ]

    for head_idx in range(1, 6):
        num_anchor_per_pixel = 1
        if anchor_sizes is not None:
            num_anchor_per_pixel *= len(anchor_sizes[head_idx])
        if anchor_size_ratio is not None:
            num_anchor_per_pixel *= len(anchor_size_ratio[head_idx])

        act_loc_layers += [nn.ReLU(False)]
        act_conf_layers += [nn.ReLU(False)]
        loc_layers += [
            nn.Conv2d(
                int(256 * width_mult * CM_IN_FPN),
                4 * num_anchor_per_pixel,
                kernel_size=1,
                padding=0,
            )
        ]
        conf_layers += [
            nn.Conv2d(
                int(256 * width_mult * CM_IN_FPN),
                num_classes * num_anchor_per_pixel,
                kernel_size=1,
                padding=0,
            )
        ]

    return (act_loc_layers, act_conf_layers, loc_layers, conf_layers)


def _forward_base(base, x):
    return base(x)


def _forward_head(
    act_loc_layers,
    act_conf_layers,
    loc_layers,
    conf_layers,
    input,
    maxout_bg_size=3,
    anchor_sizes=None,
    anchor_size_ratio=None,
):
    loc_outputs_over_stack = list()
    conf_outputs_over_stack = list()

    for stack_id in range(len(input)):
        num_anchor_per_pixel = 1
        if anchor_sizes is not None:
            num_anchor_per_pixel *= len(anchor_sizes[0])
        if anchor_size_ratio is not None:
            num_anchor_per_pixel *= len(anchor_size_ratio[0])

        loc_outputs = list()
        conf_outputs = list()

        x = input[stack_id]

        loc_x = act_loc_layers[0](x[0])
        conf_x = act_conf_layers[0](x[0])

        loc_x = loc_layers[0](loc_x)
        conf_x = conf_layers[0](conf_x)

        # for dealing with multiple anchors on each pixel
        if anchor_sizes is not None or anchor_size_ratio is not None:
            conf_x_per_pixel = []
            for i in range(num_anchor_per_pixel):
                start_idx = 4 * i
                # for dealing with maxout BG labels
                maxout_bg_conf, _ = torch.max(
                    conf_x[:, start_idx : start_idx + maxout_bg_size, :, :],
                    dim=1,
                    keepdim=True,
                )
                fg_conf = conf_x[
                    :, start_idx + maxout_bg_size : start_idx + maxout_bg_size + 1, :, :
                ]
                conf_x_per_pixel += [maxout_bg_conf, fg_conf]

            conf_x = torch.cat(conf_x_per_pixel, dim=1)
        else:
            max_conf, _ = torch.max(
                conf_x[:, 0:maxout_bg_size, :, :], dim=1, keepdim=True
            )
            conf_x = torch.cat(
                (max_conf, conf_x[:, maxout_bg_size : maxout_bg_size + 1, :, :]), dim=1
            )

        loc_outputs.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf_outputs.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(x)):
            conf_outputs.append(
                conf_layers[i](act_conf_layers[i](x[i]))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                loc_layers[i](act_loc_layers[i](x[i])).permute(0, 2, 3, 1).contiguous()
            )

        conf_outputs_over_stack.append(conf_outputs)
        loc_outputs_over_stack.append(loc_outputs)

    return (conf_outputs_over_stack, loc_outputs_over_stack)
