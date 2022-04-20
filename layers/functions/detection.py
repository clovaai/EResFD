"""
Code Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/functions/detection.py
"""
# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch

from ..bbox_utils import decode, nms
import torch.nn as nn
import os

from misc.misc import Stopwatch


class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg, log_file_path=None):
        super(Detect, self).__init__()
        self.num_classes = cfg.NUM_CLASSES
        self.conf_thresh = cfg.CONF_THRESH
        self.nms_thresh = cfg.NMS_THRESH
        self.nms_top_k = cfg.NMS_TOP_K
        self.top_k = cfg.TOP_K
        self.variance = cfg.VARIANCE

        print("Detect: self.thresh: %f" % self.conf_thresh)
        print("Detect: self.nms_thresh(IOU-overlap th.): %f" % self.nms_thresh)
        print("Detect: self.nms_top_k: %f" % self.nms_top_k)
        print("Detect: self.top_k: %f" % self.top_k)

        # NOTE: time check for the NMS
        self.timer = Stopwatch("NMS")
        self.avg_rejected = 0
        self.avg_survived = 0
        self.avg_rejection_rate = 0.0
        self.avg_elapsed_nms = 0.0
        self.counter = 0
        if log_file_path is not None:
            self.log_file = open(log_file_path + "detect_log.txt", "w")
        else:
            self.log_file = None

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4] or [1, num_priors, 4]
        """
        # get batch_size
        num = loc_data.size(0)
        # get # of prior boxes
        if prior_data.dim() == 3:
            num_priors = prior_data.size(1)
        else:
            num_priors = prior_data.size(0)

        # import pdb; pdb.set_trace()
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        if prior_data.dim() == 3:
            batch_priors = prior_data.view(-1, 4)
        else:
            # NOTE: expand 1 X #priors X 4 to batch_size X #priors X 4
            # if batch_size(i.e. num) is 1, expand operation can be ommited
            batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
            batch_priors = batch_priors.contiguous().view(-1, 4)

        # NOTE: dim. conf_prds: batch_size, #classes, #priors
        # NOTE: dim. batch_priors: batch_size(expanded) X #priors, 4

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        # set memory
        if self.top_k != -1:
            output = torch.zeros(num, self.num_classes, self.top_k, 5)
        else:
            output = None

        # NOTE: for each sample (in a batch)
        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            # NOTE: do not care in case where cl is 0 (due to zero is the background-idx)
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]

                total_boxes = conf_scores[cl].size(0)
                survived_boxes = c_mask.sum().item()
                rejected_boxes = total_boxes - survived_boxes
                rejection_rate = rejected_boxes * 1.0 / total_boxes

                self.avg_survived += survived_boxes
                self.avg_rejected += rejected_boxes
                self.avg_rejection_rate += rejection_rate
                self.counter += 1

                print(
                    "th: %f, rejection rate: %f(%f) %d/%d, survived: %d(%f)"
                    % (
                        self.conf_thresh,
                        rejection_rate,
                        self.avg_rejection_rate / self.counter,
                        rejected_boxes,
                        total_boxes,
                        survived_boxes,
                        self.avg_survived * 1.0 / self.counter,
                    )
                )
                if self.log_file is not None:
                    self.log_file.write(
                        "th: %f, rejection rate: %f(%f) %d/%d, survived: %d(%f)\n"
                        % (
                            self.conf_thresh,
                            rejection_rate,
                            self.avg_rejection_rate / self.counter,
                            rejected_boxes,
                            total_boxes,
                            survived_boxes,
                            self.avg_survived * 1.0 / self.counter,
                        )
                    )

                # original
                # if scores.dim() == 0:
                #    continue

                # NOTE: modified
                if torch.sum(c_mask).item() == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)

                # print('# boxes before NMS: %d' % boxes.size(0))

                # NOTE: dim. boxes_: #masked_priors, 4
                #      dim. scores: #masked_priors
                # import pdb; pdb.set_trace()
                with self.timer:
                    ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
                self.avg_elapsed_nms += self.timer.latency
                print(
                    "elapsed (NMS): %f(%f), survived: %d(%f)"
                    % (
                        self.timer.latency,
                        self.avg_elapsed_nms / self.counter,
                        survived_boxes,
                        self.avg_survived / self.counter,
                    )
                )
                if self.log_file is not None:
                    self.log_file.write(
                        "elapsed (NMS): %f(%f), survived: %d(%f)\n"
                        % (
                            self.timer.latency,
                            self.avg_elapsed_nms / self.counter,
                            survived_boxes,
                            self.avg_survived / self.counter,
                        )
                    )

                if self.top_k != -1:
                    count = count if count < self.top_k else self.top_k
                    output[i, cl, :count] = torch.cat(
                        (scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1
                    )
                else:
                    # TODO: num of class > 2, batch size > 1
                    output = torch.zeros(num, self.num_classes, len(scores[ids]), 5)
                    output[i, cl] = torch.cat(
                        (scores[ids].unsqueeze(1), boxes_[ids]), 1
                    )

        return output
