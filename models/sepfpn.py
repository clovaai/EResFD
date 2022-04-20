"""
EResFD
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""
import torch
import os
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, use_bn=True, use_relu=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2
        if use_bn:
            if use_relu:
                self.conv = nn.Sequential(
                    nn.Conv2d(
                        in_c,
                        out_c,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(True),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(
                        in_c,
                        out_c,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_c),
                )
        else:
            if use_relu:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding),
                    nn.ReLU(True),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding)
                )

    def forward(self, x):
        return self.conv(x)


class FullySepFPN(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        sep_idx=3,
        first_lateral_in_c=None,
        use_bn=False,
        use_relu=False,
        relu_after_aggregate=False,
    ):
        super(FullySepFPN, self).__init__()
        self.init = 0.5
        self.eps = 0.0001
        self.itm_idx = [1, 2, 3, 4]
        self.origin_idx = [1, 2, 3, 4]
        self.sep_idx = sep_idx
        self.relu_after_aggregate = relu_after_aggregate

        if first_lateral_in_c is None:
            first_lateral_in_c = in_c

        laterals = [
            Conv(first_lateral_in_c, out_c, 1, 1, use_bn=use_bn, use_relu=use_relu)
        ] + [
            Conv(in_c, out_c, 1, 1, use_bn=use_bn, use_relu=use_relu) for _ in range(4)
        ]
        self.laterals = nn.ModuleList(laterals)

        self.itm = nn.ModuleList(
            [
                Conv(out_c, out_c, 3, 1, use_bn=use_bn, use_relu=use_relu)
                for _ in range(4)
            ]
        )

        self.w1 = nn.Parameter(torch.Tensor(2, 9).fill_(self.init))
        self.relu1 = nn.ReLU()
        self.log_idx = 0
        if self.training:
            if not os.path.exists("feature_fusion_weights"):
                os.makedirs("feature_fusion_weights")

    def _td_aggregate(self, lateral, upsample_target, weight, origin=None):
        _, _, H, W = lateral.size()
        # assert H == upsample_target.size(2) * 2 and  W == upsample_target.size(3) * 2
        upsample_target = F.interpolate(upsample_target, size=(H, W), mode="nearest")

        if origin is not None:
            out = (
                weight[0] * lateral + weight[1] * upsample_target + weight[2] * origin
            ) / (weight[0] + weight[1] + weight[2] + self.eps)
            if self.relu_after_aggregate:
                return F.relu(out, True)
            else:
                return out
        else:
            out = (weight[0] * lateral + weight[1] * upsample_target) / (
                weight[0] + weight[1] + self.eps
            )
            if self.relu_after_aggregate:
                return F.relu(out, True)
            else:
                return out

    def _aggregate(self, lateral, origin, weight):
        out = (weight[0] * lateral + weight[1] * origin) / (
            weight[0] + weight[1] + self.eps
        )
        if self.relu_after_aggregate:
            return F.relu(out, True)
        else:
            return out

    def forward(self, sources):
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.eps  # normalize

        if self.training:
            # save to local docker
            # torch.save(
            #     w1,
            #     "feature_fusion_weights/feature_fusion_weights_it%d.pt" % self.log_idx,
            # )
            self.log_idx += 1

        # input origins
        inp_org_blobs = [
            source.clone()
            for idx, source in enumerate(sources)
            if idx in self.origin_idx
        ]

        # lateral
        laterals = [self.laterals[idx](fm) for idx, fm in enumerate(sources[:-1])]
        laterals.append(sources[-1])

        # high level top-down & lateral aggregate
        h_td_paths = [laterals[-1]]
        for i in range(
            5, self.sep_idx, -1
        ):  # NOTE: sep_idx==2: i={5,4,3}, sep_idx==3: i={5,4}, sep_idx==4: i={5}
            h_td_paths.insert(
                0, self._td_aggregate(laterals[i - 1], h_td_paths[0], w1[:, i - 1])
            )  # sep_idx==2: w1={4,3,2}, sep_idx==3: w1={4,3}, sep_idx==4: w1={4}

        # low level top-down & lateral aggregate
        l_td_paths = []
        for i in range(
            self.sep_idx, 0, -1
        ):  # NOTE: sep_idx==2: i={2,1}, sep_idx==3: i={3,2,1}, sep_idx==4: i={4,3,2,1}
            if i == self.sep_idx:
                # NOTE: inp_org_blobs sanity check passed (no in-place operations in _td_aggregate)
                l_td_paths.insert(
                    0,
                    self._td_aggregate(
                        laterals[i - 1], inp_org_blobs[i - 1], w1[:, i - 1]
                    ),
                )  # sep_idx==2: w1={1}, sep_idx==3: w1={2}, sep_idx==4: w1={3}
            else:
                l_td_paths.insert(
                    0, self._td_aggregate(laterals[i - 1], l_td_paths[0], w1[:, i - 1])
                )  # sep_idx==2: w1={0}, sep_idx==3: w1={1,0}, sep_idx==4: w1={2,1,0}

        # low level intermediate
        for i in range(
            1, self.sep_idx
        ):  # NOTE: sep_idx==2: i={1}, sep_idx==3: i={1,2}, sep_idx==4: i={1,2,3}
            l_td_paths[i] = self.itm[i - 1](l_td_paths[i])

        # low level post aggregate
        for i in range(
            1, self.sep_idx
        ):  # NOTE: sep_idx ==2: i={1}, sep_idx==3: i={1,2}, sep_idx==4: i={1,2,3}
            l_td_paths[i] = self._aggregate(
                l_td_paths[i], inp_org_blobs[i - 1], w1[:, i + 4]
            )  # NOTE: sep_idx==2: w1={5}, sep_idx==3: w1={5,6}, sep_idx==4: w1={5,6,7}

        # high level intermediate & post aggregate
        h_td_paths = [
            self._aggregate(
                self.itm[idx + self.sep_idx - 1](td),
                inp_org_blobs[idx + self.sep_idx - 1],
                w1[:, idx + self.sep_idx + 4],
            )
            if idx + self.sep_idx in self.itm_idx[self.sep_idx - 1 :]
            else td
            for idx, td in enumerate(h_td_paths)
        ]

        return l_td_paths + h_td_paths
