# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from models.eresfd import build_model
from utils.augmentations import to_chw_bgr

parser = argparse.ArgumentParser(description="pyramidbox demo")
parser.add_argument(
    "--save_dir", type=str, default="tmp/", help="Directory for detect result"
)
parser.add_argument(
    "--model", type=str, default="weights/eresfd_16.pth", help="trained model"
)
parser.add_argument(
    "--thresh", default=0.4, type=float, help="Final confidence threshold"
)
parser.add_argument("--width_mult", default=0.0625, type=float, help="width-multiplier")
parser.add_argument(
    "--anchor_steps",
    type=int,
    nargs="+",
    default=cfg.STEPS,
    help="anchor stride settings. default is [4, 8, 16, 32, 64, 128]",
)
parser.add_argument(
    "--anchor_sizes",
    type=int,
    nargs="+",
    default=[16, 32, 64, 128, 256, 512],
    help="anchor size settings. default is [16, 32, 64, 128, 256, 512]",
)
parser.add_argument(
    "--anchor_scales",
    type=float,
    nargs="+",
    default=[1.0],
    action="append",
    help="anchor size scales per location. default is 1",
)
parser.add_argument(
    "--anchor_size_ratio",
    type=float,
    nargs="+",
    default=[1.25],
    action="append",
    help="anchor size ratio. default is 1.",
)
args = parser.parse_args()

# anchor settings
# if anchor scales are same along the feature maps
if len(args.anchor_scales) == 1:
    args.anchor_scales = args.anchor_scales[0]

    anchors = []
    for anchor_size in args.anchor_sizes:
        anchors_per_size = []
        if isinstance(args.anchor_scales, list):
            for anchor_scale in args.anchor_scales:
                anchors_per_size.append(anchor_size * anchor_scale)
        else:
            anchors_per_size.append(anchor_size)
        anchors.append(anchors_per_size)
else:
    anchors = []
    for layer_idx, anchor_size in enumerate(args.anchor_sizes):
        try:
            anchor_scales = args.anchor_scales[layer_idx + 1]
        except IndexError as e:
            pass

        anchors_per_size = []
        for anchor_scale in anchor_scales:
            anchors_per_size.append(anchor_size * anchor_scale)
        anchors.append(anchors_per_size)

cfg.ANCHOR_SIZES = anchors
# # if anchor size ratio is given as default value (i.e. 1)
if len(args.anchor_size_ratio) == 1:
    args.anchor_size_ratio = [[el] for el in args.anchor_size_ratio]
    args.anchor_size_ratio = args.anchor_size_ratio * 6
else:
    args.anchor_size_ratio = args.anchor_size_ratio[1:]

    # if anchor size ratio is same along the detection layers
    if len(args.anchor_size_ratio) == 1:
        args.anchor_size_ratio = [el for el in args.anchor_size_ratio]
        args.anchor_size_ratio = args.anchor_size_ratio * 6

    # if anchor size ratio is different along the detection layers
    else:
        assert len(args.anchor_size_ratio) == 6


cfg.ANCHOR_SIZE_RATIO = args.anchor_size_ratio
print("ANCHOR ratio settings: ", cfg.ANCHOR_SIZE_RATIO)

cfg.STEPS = args.anchor_steps
print("ANCHOR stride settings: ", cfg.STEPS)


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


def detect(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == "L":
        img = img.convert("RGB")

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(1200 * 1100 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(
        img,
        None,
        None,
        fx=max_im_shrink,
        fy=max_im_shrink,
        interpolation=cv2.INTER_LINEAR,
    )

    x = to_chw_bgr(image)
    x = x.astype("float32")
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]
    # x = x * cfg.scale

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.2f}".format(score)
            text_size, baseline = cv2.getTextSize(
                conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1
            )
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(
                img,
                (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                (p1[0] + text_size[0], p1[1] + text_size[1]),
                [255, 0, 0],
                -1,
            )
            cv2.putText(
                img,
                conf,
                (p1[0], p1[1] + baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                8,
            )

    t2 = time.time()
    print("detect:{} timer:{}".format(img_path, t2 - t1))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


if __name__ == "__main__":
    net = build_model("test", cfg.NUM_CLASSES, args.width_mult)
    net.load_weights(args.model)
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_path = "./img"
    img_list = [
        os.path.join(img_path, x) for x in os.listdir(img_path) if x.endswith("jpg")
    ]
    for path in img_list:
        with torch.no_grad():
            detect(net, path, args.thresh)
