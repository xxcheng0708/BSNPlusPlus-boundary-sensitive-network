# -*- coding: utf-8 -*-
"""
    UNet部分代码参考：https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py
    Non-Local部分代码参考：https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.4.1_to_1.1.0/lib/non_local_dot_product.py
    BMN部分代码参考：https://github.com/JJBOY/BMN-Boundary-Matching-Network
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    """
    BSN++中CBG模块的UNet的每个单元unit
    """

    def __init__(self, in_ch, out_ch, is_output=False):
        super(ConvUnit, self).__init__()
        module_list = [nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)]
        if is_output is False:
            module_list.append(nn.BatchNorm1d(out_ch))
            module_list.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*module_list)

    def forward(self, x):
        x = self.conv(x)
        return x


class NestedUNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=400, out_ch=2):
        super(NestedUNet, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2)

        n1 = 512
        filters = [n1, n1 * 2, n1 * 3]
        # UNet的第一列
        self.conv0_0 = ConvUnit(in_ch, filters[0], is_output=False)
        self.conv1_0 = ConvUnit(filters[0], filters[0], is_output=False)
        self.conv2_0 = ConvUnit(filters[0], filters[0], is_output=False)

        # UNet的第二列
        self.conv0_1 = ConvUnit(filters[1], filters[0], is_output=False)
        self.conv1_1 = ConvUnit(filters[1], filters[0], is_output=False)

        # UNet的第三列
        self.conv0_2 = ConvUnit(filters[2], filters[0], is_output=False)

        # 输出，红点位置
        self.final = nn.Conv1d(filters[0] * 3, out_ch, kernel_size=1)
        # self.final = ConvUnit(filters[0] * 3, out_ch, is_output=True)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # 同步输出正向和反向的中间特征，用于计算MSELoss
        out_feature = torch.cat([x0_0, x0_1, x0_2], 1)
        final_feature = self.final(out_feature)
        out = self.out(final_feature)

        return out, out_feature


class PositionAwareAttentionModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=None, dimension=2):
        super(PositionAwareAttentionModule, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dimension = dimension

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if self.dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2,))
            bn = nn.BatchNorm1d

        self.g = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.phi = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        if self.sub_sample:
            # 对g和phi进行相同的下采样可以进一步降低计算量
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)
        # value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=2)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)

        z = y + x
        return z


class ChannelAwareAttentionModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2):
        super(ChannelAwareAttentionModule, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dimension = dimension

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if self.dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2,))
            bn = nn.BatchNorm1d

        self.g = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.phi = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=2)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)

        z = y + x
        return z


def conv_block(in_ch, out_ch, kernel_size=3, stride=1, bn_layer=False, activate=False):
    module_list = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=1)]
    if bn_layer:
        module_list.append(nn.BatchNorm2d(out_ch))
        module_list.append(nn.ReLU(inplace=True))
    if activate:
        module_list.append(nn.Sigmoid())
    conv = nn.Sequential(*module_list)
    return conv


class ProposalRelationBlock(nn.Module):
    def __init__(self, in_channels, inter_channles=128, sub_sample=False):
        super(ProposalRelationBlock, self).__init__()
        self.p_net = PositionAwareAttentionModule(in_channels, inter_channels=inter_channles, sub_sample=sub_sample, dimension=2)
        self.c_net = ChannelAwareAttentionModule(in_channels, inter_channels=inter_channles, dimension=2)
        self.conv0_0 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv0_1 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)

        self.conv1 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv2 = conv_block(in_channels, 2, 3, 1, bn_layer=False, activate=True)
        self.conv3 = conv_block(in_channels, 2, 3, 1, bn_layer=False, activate=True)
        self.conv4 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv5 = conv_block(in_channels, 2, 3, 1, bn_layer=False, activate=True)

    def forward(self, x):
        x_p = self.conv0_0(x)
        x_c = self.conv0_1(x)

        x_p = self.p_net(x_p)
        x_c = self.c_net(x_c)

        x_p_0 = self.conv1(x_p)
        x_p_1 = self.conv2(x_p_0)

        x_c_0 = self.conv4(x_c)
        x_c_1 = self.conv5(x_c_0)

        x_p_c = self.conv3(x_p_0 + x_c_0)

        # x_out = (x_p_1 + x_c_1 + x_p_c) / 3
        return x_p_1, x_c_1, x_p_c


class BSNPP(nn.Module):
    def __init__(self, opt):
        super(BSNPP, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.prop_boundary_ratio = opt["prop_boundary_ratio"]
        self.num_sample = opt["num_sample"]
        self.num_sample_perbin = opt["num_sample_perbin"]
        self.feat_dim = opt["feat_dim"]

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm1d(self.hidden_dim_1d),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm1d(self.hidden_dim_1d),
            nn.ReLU(inplace=True)
        )

        # Complementary Boundary Generator
        self.x_1d_cbg = NestedUNet(in_ch=self.hidden_dim_1d, out_ch=2)

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim_1d),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),
                      stride=(self.num_sample, 1, 1)),
            nn.BatchNorm3d(self.hidden_dim_3d),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim_2d),
            nn.ReLU(inplace=True)
        )
        self.proposal_block = ProposalRelationBlock(self.hidden_dim_2d, self.hidden_dim_2d, sub_sample=True)

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        cbg_prob, cbg_feature = self.x_1d_cbg(base_feature)
        start = cbg_prob[:, 0, :].squeeze(1)
        end = cbg_prob[:, 1, :].squeeze(1)
        # print("start: {}, end: {}".format(start.shape, end.shape))

        confidence_map = self.x_1d_p(base_feature)
        # print("confidence_map: {}".format(confidence_map.shape))
        confidence_map = self._boundary_matching_layer(confidence_map)
        # print("confidence_map: {}".format(confidence_map.shape))
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        confidence_map_p, confidence_map_c, confidence_map_p_c = self.proposal_block(confidence_map)
        return confidence_map_p, confidence_map_c, confidence_map_p_c, start, end, cbg_feature

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0], input_size[1], self.num_sample, self.tscale,
                                                        self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)
        print("weight matrix:", self.sample_mask.shape)


if __name__ == '__main__':
    import opts

    opt = opts.parse_opt()
    opt = vars(opt)

    # cbg_net = NestedUNet(in_ch=400, out_ch=2)
    # p_net = PositionAwareAttentionModule(256, 512, sub_sample=True)
    # c_net = ChannelAwareAttentionModule(256, 512, sub_sample=False)
    # prb_net = ProposalRelationBlock(256, 512, sub_sample=True)

    x = torch.randn(2, 400, 100)
    x_prb = torch.randn(2, 256, 100, 100)

    # print("#" * 40, "CBG", "#" * 40)
    # cbg_out, cbg_feature = cbg_net(x)
    # print(cbg_out.shape, cbg_feature.shape)
    #
    # print("#" * 40, "PositionAttention", "#" * 40)
    # p_out = p_net(x_prb)
    # print(p_out.shape)
    #
    # print("#" * 40, "ChannelAttention", "#" * 40)
    # c_out = c_net(x_prb)
    # print(c_out.shape)
    #
    # print("#" * 40, "PRB", "#" * 40)
    # prb_out = prb_net(x_prb)
    # print(prb_out.shape)

    print("#" * 40, "BSNPP", "#" * 40)
    bsnpp_net = BSNPP(opt)
    confidence_map_p, confidence_map_c, confidence_map_p_c, start, end, feature = bsnpp_net(x)
    print(confidence_map_p.shape)
