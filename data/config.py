# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
from easydict import EasyDict
import numpy as np


_C = EasyDict()
cfg = _C
# data augument config
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 18
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 6.0
_C.apply_distort = True
_C.apply_expand = False
_C.img_mean = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype(
    "float32"
)  # order in BGR
_C.resize_width = 640
_C.resize_height = 640
_C.scale = 1 / 127.0
_C.anchor_sampling = True
_C.filter_min_face = True

# train config
# _C.LR_STEPS = (120, 198, 250)
# _C.MAX_STEPS = 200000 # wider, vgg, se-resnet bs30
_C.MAX_STEPS = 280000  # hg, bs16
# _C.LR_STEPS = (80000, 100000, 120000) #wider, vgg, se-resnet bs30
# _C.LR_STEPS = (120000,150000, 180000) #wider, bs16
_C.LR_STEPS = (150000, 180000, 200000)  # wider, bs16
_C.EPOCHES = 30000000

# used for resume
_C.START_EPOCHES = 0
_C.ITERATIONS = 0

# anchor config
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]  # feature-map size
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]  # stride from size of 640 input
_C.ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]  # scale
_C.ANCHOR_SIZE_RATIO = 1.0
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# detection config
_C.NMS_THRESH = 0.3  # IoU overlap threshold
_C.NMS_TOP_K = 5000
_C.TOP_K = 750
_C.CONF_THRESH = 0.05  # prior to NMS in detection.py

# loss config
_C.NEG_POS_RATIOS = 3
_C.NUM_CLASSES = 2
_C.USE_NMS = True

# Maxout Background label
_C.MAXOUT_BG_SIZE = 3
# Maxin Foreground label
_C.MAXIN_FG_SIZE = 1

# face config
_C.FACE = EasyDict()
_C.FACE.TRAIN_FILE = "widerface_train.txt"
_C.FACE.VAL_FILE = "widerface_val.txt"
_C.FACE.FDDB_DIR = "/home/storage/fddb"
_C.FACE.WIDER_DIR = "/home/storage/widerface"
_C.FACE.OVERLAP_THRESH = [0.1, 0.35, 0.5]


# FOCAL LOSS
_C.FOCAL_LOSS_ALPHA = 0.25
_C.FOCAL_LOSS_GAMMA = 2.0

# target face bbox clipping
_C.TARGET_FACE_BBOX_CLIPPING = False
