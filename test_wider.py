# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import os.path as osp

import time
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from data.config import cfg
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr
from eval_tools.evaluation import evaluation
from models.eresfd import build_model

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


def image_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def detect_face(net, img, shrink, thresh=0.05):
    # img size가 1700 x 1200 보다 크다면, 1700x1200로 shrink
    if shrink != 1:
        img = cv2.resize(
            img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR
        )
        min_size, min_axis = np.min(img.shape[:2]), np.argmin(img.shape[:2])
        # NOTE: set minimum size
        if min_size < 129:
            width, height = None, None
            if min_axis == 0:
                height = 128
            if min_axis == 1:
                width = 128
            img = image_resize(img, width=width, height=height)

    x = to_chw_bgr(img)
    x = x.astype("float32")
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]  # to rgb

    x = Variable(torch.from_numpy(x).unsqueeze(0), requires_grad=False)

    if use_cuda:
        x = x.cuda()

    with torch.no_grad():
        y = net(x)  # decoded y values.
    detections = y.data.cpu().numpy()

    det_conf = detections[0, 1, :, 0]
    det_xmin = img.shape[1] * detections[0, 1, :, 1] / shrink
    det_ymin = img.shape[0] * detections[0, 1, :, 2] / shrink
    det_xmax = img.shape[1] * detections[0, 1, :, 3] / shrink
    det_ymax = img.shape[0] * detections[0, 1, :, 4] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= thresh)[
        0
    ]  # NOTE: this is duplicated since detect.py does confidence filtering before NMS.

    det = det[keep_index, :]

    return det


def flip_test(net, image, shrink, thresh=0.05):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink, thresh=thresh)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(net, image, max_im_shrink, thresh=0.05):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st, thresh=thresh)
    index = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30
    )[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt, thresh=thresh)

    # enlarge small image x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(
            np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
            < 100
        )[0]
        det_b = det_b[index, :]
    else:
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
            > 30
        )[0]
        det_b = det_b[index, :]

    return det_s, det_b


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        # NOTE: modified
        if merge_index.shape[0] > 1:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
                det_accu[:, -1:]
            )
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum
        else:
            try:
                # if dets already exist
                dets = np.row_stack((dets, det_accu))
            except:
                # if dets is not decleared anywhere
                dets = det_accu

    try:
        dets = dets[0 : cfg.TOP_K, :]
    except:
        # NOTE if dets.size(0) is less than cfg.TOP_K
        dets = None

    return dets


def get_data():
    subset = "val"
    if subset is "val":
        wider_face = sio.loadmat("./eval_tools/wider_face_val_file_list.mat")
    else:
        wider_face = sio.loadmat("./eval_tools/wider_face_test_file_list.mat")
    event_list = wider_face["event_list"]
    file_list = wider_face["file_list"]
    del wider_face

    imgs_path = os.path.join(cfg.FACE.WIDER_DIR, "WIDER_{}".format(subset), "images")

    return event_list, file_list, imgs_path


def predict_wider(
    net, thresh=0.05, exp="wider_eval", single_scale_test=False, no_max_im_shrink=False
):
    event_list, file_list, imgs_path = get_data()
    cfg.USE_NMS = True

    counter = 0
    for index, event in enumerate(event_list):
        filelist = file_list[index][0]
        path = os.path.join(exp, event[0][0])
        if not os.path.exists(path):
            os.makedirs(path)

        for num, file in enumerate(filelist):
            im_name = file[0][0]
            in_file = os.path.join(imgs_path, event[0][0], im_name[:] + ".jpg")
            img = Image.open(in_file)
            if img.mode == "L":
                img = img.convert("RGB")
            img = np.array(img)

            max_im_shrink = np.sqrt(1700 * 1200 / (img.shape[0] * img.shape[1]))

            if no_max_im_shrink:
                shrink = 1
            else:
                # 1700 * 1200 보다 크면, 1700 * 1200 로 shrink
                shrink = max_im_shrink if max_im_shrink < 1 else 1

            counter += 1

            t1 = time.time()

            # NOTE: modified
            # det0 shape: 750, 5
            det0 = detect_face(net, img, shrink, thresh=thresh)

            if not single_scale_test:
                det1 = flip_test(net, img, shrink, thresh=thresh)  # flip test
                [det2, det3] = multi_scale_test(net, img, max_im_shrink, thresh=thresh)

                det = np.row_stack((det0, det1, det2, det3))
                dets = bbox_vote(det)
            else:
                dets = det0

            t2 = time.time()
            print("Detect %04d th image costs %.4f" % (counter, t2 - t1), flush=True)

            fout = open(osp.join(exp, event[0][0], im_name + ".txt"), "w")
            fout.write("{:s}\n".format(event[0][0] + "/" + im_name + ".jpg"))

            if dets is not None:
                fout.write("{:d}\n".format(dets.shape[0]))
                for i in range(dets.shape[0]):
                    xmin = dets[i][0]
                    ymin = dets[i][1]
                    xmax = dets[i][2]
                    ymax = dets[i][3]
                    score = dets[i][4]
                    fout.write(
                        "{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n".format(
                            xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score
                        )
                    )
            else:
                fout.write("0\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="widerface evaluation")
    parser.add_argument("--model", type=str, default="", help="trained model")
    parser.add_argument("--wider_root", default="/Users/user/works/widerface")
    parser.add_argument(
        "--width_mult", default=0.0625, type=float, help="width-multiplier"
    )

    parser.add_argument(
        "--legend_name", type=str, default="Ours", help="name of method"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="./eval_tools/wider_ground_truth",
        help="wider gt",
    )

    parser.add_argument(
        "--exp", type=str, default="prediction_wider_for_eval", help="output path"
    )
    parser.add_argument(
        "--thresh", default=0.05, type=float, help="Final confidence threshold"
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="box overlap threshold between predicted bbox and gt bbox",
    )

    parser.add_argument(
        "--nms_top_k", default=5000, type=int, help="# of top k boxes prior to nms"
    )
    parser.add_argument(
        "--top_k", default=750, type=int, help="# of top k boxes after nms"
    )
    parser.add_argument(
        "--nms_thresh", default=0.3, type=float, help="IOU overlap in the NMS"
    )
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

    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="if we do not need to predict bboxes again",
    )
    parser.add_argument("--pause", default=0, type=int, help="")
    parser.add_argument("--iteration", default=0, type=int, help="")
    parser.add_argument("--session_name", default=0, type=str, help="")

    parser.add_argument(
        "--single_scale_test",
        action="store_true",
        help="single scale test. default: False",
    )
    parser.add_argument(
        "--no_max_im_shrink",
        action="store_true",
        help="not using max img shrink. default: False (Use max im shrink)",
    )

    parser.add_argument("--detect_log_write", action="store_true")

    args = parser.parse_args()

    cfg.FACE.WIDER_DIR = args.wider_root

    print(args)

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

    cfg.CONF_THRESH = args.thresh
    # NMS overlap thresh
    cfg.NMS_THRESH = args.nms_thresh
    cfg.NMS_TOP_K = args.nms_top_k
    cfg.TOP_K = args.top_k

    if not args.eval_only:
        net = build_model("test", cfg.NUM_CLASSES, width_mult=args.width_mult)
        print(net)
        net.load_weights(args.model)
        print("model loading done, %s" % args.model)
        net.eval()

        if use_cuda:
            net.cuda()
            cudnn.benckmark = True

        # save predicted bboxes for each image in txt format
        predict_wider(
            net, args.thresh, args.exp, args.single_scale_test, args.no_max_im_shrink
        )
        print("Prediction done for widerface")

    # compute mAP
    # import pdb; pdb.set_trace()
    evaluation(
        pred=args.exp,
        gt_path=args.gt_path,
        iou_thresh=args.iou_thresh,
        plt_filename=args.exp + "_pr-curve",
        legend_name=args.legend_name,
    )
    print("Evaluation done in %s" % args.exp)
