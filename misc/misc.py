import os
import sys
import time
import math

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from thop import profile


class Stopwatch:
    def __init__(self, title, silance=True):
        self.title = title
        self.silance = silance

    def __enter__(self):
        self.t0 = time.time()
        # logging.debug('{} begin'.format(self.title))

    def __exit__(self, type, value, traceback):
        current_time = time.time()
        if not self.silance:
            print("{} : {}ms".format(self.title, int((current_time - self.t0) * 1000)))
        self.latency = current_time - self.t0


def check_model_complexity(m):
    num_params = get_num_params(m)
    size_params = get_size_params_in_MB(num_params)
    return num_params, size_params


def get_num_params(model):
    num_params = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            params = np.prod(m.weight.size())
            num_params += params
            if m.bias is not None:
                num_params += np.prod(m.bias.size())
        elif isinstance(m, nn.BatchNorm2d):
            pass

    return num_params


def get_size_params_in_MB(num_params):
    size_bytes = num_params * 4
    return size_bytes / 1024.0 / 1024.0


def check_detector_size(model):
    t_num_params, t_size_params = 0, 0
    module_name = ["base", "loc", "conf"]
    for idx, module in enumerate(model):
        num_params, size_params = check_model_complexity(module)
        print(
            "%s: num_params, size_params: %.4f M, %f MB"
            % (module_name[idx], num_params / 1000000, size_params)
        )
        t_num_params += num_params
        t_size_params += size_params
    print("total: num_params, size_params: %f, %f MB" % (t_num_params, t_size_params))

    return (t_num_params, t_size_params)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
