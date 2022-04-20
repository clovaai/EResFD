"""
EResFD
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""
# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from layers.functions.prior_box import _get_prior_box
from layers.functions.detection import Detect
from data.config import cfg
from models.eresfd_16 import (
    _get_base_layers,
    _get_head_layers,
    _forward_base,
    _forward_head,
)


def build_model(phase, num_classes=2, width_mult=1.0, weights_init=None):

    base_ = _get_base_layers(width_mult=width_mult)

    # if multiple anchor scales per pixel location
    if any(
        isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
        for anchors_per_layer in cfg.ANCHOR_SIZES
    ):
        # if multiple anchor size ratio per pixel location
        if any(
            isinstance(anchor_size_ratio_per_layer, list)
            and len(anchor_size_ratio_per_layer) > 1
            for anchor_size_ratio_per_layer in cfg.ANCHOR_SIZE_RATIO
        ):
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_sizes=cfg.ANCHOR_SIZES,
                anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
            )
        else:
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_sizes=cfg.ANCHOR_SIZES,
            )
    else:
        if any(
            isinstance(anchor_size_ratio_per_layer, list)
            and len(anchor_size_ratio_per_layer) > 1
            for anchor_size_ratio_per_layer in cfg.ANCHOR_SIZE_RATIO
        ):
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
            )
        else:
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
            )

    return EResFD(
        phase,
        base_,
        head_,
        num_classes,
        _forward_base,
        _forward_head,
        anchor_steps=cfg.STEPS,
        anchor_sizes=cfg.ANCHOR_SIZES,
        anchor_clip=cfg.CLIP,
        anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
        weights_init=weights_init,
    )


class EResFD(nn.Module):
    def __init__(
        self,
        phase,
        base,
        head,
        num_classes,
        _forward_base,
        _forward_head,
        anchor_steps=cfg.STEPS,
        anchor_sizes=cfg.ANCHOR_SIZES,
        anchor_clip=cfg.CLIP,
        anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
        weights_init=None,
    ):
        super(EResFD, self).__init__()

        self.phase = phase
        self.num_classes = num_classes

        # base
        self.base = base

        # head
        self.act_loc = nn.ModuleList(head[0])
        self.act_conf = nn.ModuleList(head[1])
        self.loc = nn.ModuleList(head[2])
        self.conf = nn.ModuleList(head[3])

        if self.phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

        self._forward_base = _forward_base
        self._forward_head = _forward_head

        self.anchor_steps = anchor_steps
        self.anchor_sizes = anchor_sizes
        self.anchor_size_ratio = anchor_size_ratio
        self.anchor_clip = anchor_clip

        if any(
            isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
            for anchors_per_layer in self.anchor_sizes
        ):
            self.multiple_anchorscales_per_layer = True
        else:
            self.multiple_anchorscales_per_layer = False

        if any(
            isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
            for anchors_per_layer in self.anchor_size_ratio
        ):
            self.multiple_anchorsratios_per_layer = True
        else:
            self.multiple_anchorsratios_per_layer = False

        self.weights_init_type = weights_init

    def forward(self, x):
        # NOTE:
        # sources: a list of six [b, c, h, w] tensors
        sources = self._forward_base(self.base, x)

        # NOTE:
        # conf: a list of siz [b, h, w, num_class] tensors
        # loc:  a list of siz [b, h, w, 4] tensors
        # if multiple anchor scales per pixel location
        if self.multiple_anchorscales_per_layer:
            if self.multiple_anchorsratios_per_layer:
                conf_over_stack, loc_over_stack = self._forward_head(
                    self.act_loc,
                    self.act_conf,
                    self.loc,
                    self.conf,
                    sources,
                    anchor_sizes=self.anchor_sizes,
                    anchor_size_ratio=self.anchor_size_ratio,
                )
            else:
                conf_over_stack, loc_over_stack = self._forward_head(
                    self.act_loc,
                    self.act_conf,
                    self.loc,
                    self.conf,
                    sources,
                    anchor_sizes=self.anchor_sizes,
                )
        else:
            if self.multiple_anchorsratios_per_layer:
                conf_over_stack, loc_over_stack = self._forward_head(
                    self.act_loc,
                    self.act_conf,
                    self.loc,
                    self.conf,
                    sources,
                    anchor_size_ratio=self.anchor_size_ratio,
                )
            else:
                conf_over_stack, loc_over_stack = self._forward_head(
                    self.act_loc, self.act_conf, self.loc, self.conf, sources
                )

        # NOTE: can be precomputed
        # noting but just store spatial resolution of each feature-map
        # for example, resolution of an input is 256 by 256
        # features_maps is to be [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2]]
        features_maps = []
        for i in range(len(loc_over_stack[-1])):
            feat = []
            feat += [loc_over_stack[-1][i].size(1), loc_over_stack[-1][i].size(2)]
            features_maps += [feat]

        self.feature_maps_sizes = features_maps

        # NOTE: can be precomputed
        with torch.no_grad():
            # cfg.STEPS: [4, 8, 16, 32, 64, 128]
            # cfg.ANCHOR_SIZES: [16, 32, 64, 128, 256, 512]
            # cfg.CLIP: False
            self.priors = _get_prior_box(
                x.size()[2:],
                self.anchor_steps,
                self.anchor_sizes,
                self.anchor_clip,
                features_maps,
            )

        # concat. all features for all scale
        outputs = []
        for loc, conf in zip(loc_over_stack, conf_over_stack):
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            outputs.append((loc, conf))

        if self.phase == "test":
            loc = outputs[-1][0]
            conf = outputs[-1][1]
            output = self.detect(
                # loc preds (1, # anchors, 4)
                loc.view(loc.size(0), -1, 4),
                # conf preds (1, # anchors, 2)
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                # default box (# boxes, 4)
                # 4: [center_x, center_y, scaled_h, scaled_w]
                self.priors.type(type(x.data)),
            )
        else:
            outputs_with_priors = []
            for out in outputs:
                loc = out[0]
                conf = out[1]

                loc = loc.view(loc.size(0), -1, 4)
                conf = conf.view(conf.size(0), -1, self.num_classes)

                output = (loc, conf, self.priors)

                outputs_with_priors.append(output)

            output = outputs_with_priors

        return output

    def load_weights(self, filename):
        base = torch.load(
            filename + ".base.pth", map_location=lambda storage, loc: storage
        )
        self.base.load_state_dict(base)
        act_loc = torch.load(
            filename + ".act_loc.pth", map_location=lambda storage, loc: storage
        )
        self.act_loc.load_state_dict(act_loc)
        act_conf = torch.load(
            filename + ".act_conf.pth", map_location=lambda storage, loc: storage
        )
        self.act_conf.load_state_dict(act_conf)
        loc = torch.load(
            filename + ".loc.pth", map_location=lambda storage, loc: storage
        )
        self.loc.load_state_dict(loc)
        conf = torch.load(
            filename + ".conf.pth", map_location=lambda storage, loc: storage
        )
        self.conf.load_state_dict(conf)
        return True

    def save_weights(self, filename):
        torch.save(self.base.cpu().state_dict(), filename + ".base.pth")
        torch.save(self.act_loc.cpu().state_dict(), filename + ".act_loc.pth")
        torch.save(self.act_conf.cpu().state_dict(), filename + ".act_conf.pth")
        torch.save(self.loc.cpu().state_dict(), filename + ".loc.pth")
        torch.save(self.conf.cpu().state_dict(), filename + ".conf.pth")
        return True

    def xavier(self, param):
        init.xavier_uniform_(param)

    def kaiming(self, param):
        init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

    def normal(self, param, std=0.01):
        init.normal_(param, mean=0.0, std=std)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            if self.weights_init_type == "xavier":
                self.xavier(m.weight.data)
            elif self.weights_init_type == "kaiming":
                self.kaiming(m.weight.data)
            else:
                self.normal(m.weight.data, std=0.01)
            if m.bias is not None:
                m.bias.data.zero_()


if __name__ == "__main__":
    cfg.ANCHOR_SIZE_RATIO = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    net = build_model("train", num_classes=2, width_mult=0.0625)
    net.load_weights("./weights/eresnet_sepfpn_cpm_reproduce.pth")
    inputs = Variable(torch.randn(4, 3, 640, 640))
    output = net(inputs)
