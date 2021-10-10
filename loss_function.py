# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F


def get_mask(tscale):
    mask = np.zeros([tscale, tscale], np.float32)
    for i in range(tscale):
        for j in range(i, tscale):
            mask[i, j] = 1
    return torch.Tensor(mask)


def bi_loss(pred_score, gt_label):
    pred_score = pred_score.view(-1)
    gt_label = gt_label.view(-1)
    pmask = (gt_label > 0.5).float()
    num_entries = len(pmask)
    num_positive = torch.sum(pmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
    loss = -1 * torch.mean(loss_pos + loss_neg)
    return loss


def cbg_loss_func(pred_start, pred_end, gt_start, gt_end):
    """
    CBG模块计算开始和结束边界的概率损失
    :param pred_start:
    :param pred_end:
    :param gt_start:
    :param gt_end:
    :return:
    """
    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss = loss_start + loss_end
    return loss


def cbg_feature_loss(forward_feature, backward_feature):
    """
    计算CBG模块对于视频特征序列前向和反向计算得到的中间特征的MSE损失
    :param forward_feature:
    :param backward_feature:
    :return:
    """
    loss = F.mse_loss(forward_feature, backward_feature, reduction="mean")
    return loss


def bsnpp_pem_reg_loss_func(pred_score, gt_iou_map, mask):
    """
    PRB模块计算proposal置信度得分的回归损失
    :param pred_score:
    :param gt_iou_map:
    :param mask:
    :return:
    """
    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    weights = u_hmask + u_smmask + u_slmask

    loss = F.smooth_l1_loss(pred_score * weights, gt_iou_map * weights)
    # loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

    return loss


def bsnpp_pem_cls_loss_func(pred_score, gt_iou_map, mask):
    """
    PRB模块计算proposal置信度得分的分类损失
    :param pred_score:
    :param gt_iou_map:
    :param mask:
    :return:
    """
    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    num_positive = torch.sum(pmask)
    num_entries = num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    ratio = torch.clamp(ratio, 1.05, 21)
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss


def bsnpp_loss_func(pred_bm_p, pred_bm_c, pred_bm_p_c, pred_start, pred_end,
                    gt_iou_map, gt_start, gt_end, bm_mask,
                    pred_start_backward, pred_end_backward,
                    feature_forward, feature_backward,
                    prb_reg_weight=10.0, cbg_feature_weight=1.0, prb_weight_forward=10):
    """
    BSNPP整体损失，CBG正向损失 + CBG方向损失 + PRB正向损失 + CBG特征损失
    :param pred_bm_p:
    :param pred_bm_c:
    :param pred_bm_p_c:
    :param pred_start:
    :param pred_end:
    :param gt_iou_map:
    :param gt_start:
    :param gt_end:
    :param bm_mask:
    :param pred_start_backward:
    :param pred_end_backward:
    :param feature_forward:
    :param feature_backward:
    :param prb_reg_weight:
    :param cbg_feature_weight:
    :param prb_weight_forward:
    :return:
    """
    # pred_bm_reg = pred_bm[:, 0].contiguous()
    # pred_bm_cls = pred_bm[:, 1].contiguous()
    gt_iou_map = gt_iou_map * bm_mask

    cbg_loss_forward = cbg_loss_func(pred_start, pred_end, gt_start, gt_end)
    cbg_loss_backward = cbg_loss_func(torch.flip(pred_end_backward, dims=(1,)),
                                      torch.flip(pred_start_backward, dims=(1,)),
                                      gt_start, gt_end)

    inter_feature_loss = cbg_feature_weight * cbg_feature_loss(feature_forward, torch.flip(feature_backward, dims=(2,)))
    cbg_loss = cbg_loss_forward + cbg_loss_backward + inter_feature_loss

    prb_reg_loss_p = bsnpp_pem_reg_loss_func(pred_bm_p[:, 0].contiguous(), gt_iou_map, bm_mask)
    prb_reg_loss_c = bsnpp_pem_reg_loss_func(pred_bm_c[:, 0].contiguous(), gt_iou_map, bm_mask)
    prb_reg_loss_p_c = bsnpp_pem_reg_loss_func(pred_bm_p_c[:, 0].contiguous(), gt_iou_map, bm_mask)
    prb_reg_loss = prb_reg_weight * (prb_reg_loss_p + prb_reg_loss_c + prb_reg_loss_p_c)

    prb_cls_loss_p = bsnpp_pem_cls_loss_func(pred_bm_p[:, 1].contiguous(), gt_iou_map, bm_mask)
    prb_cls_loss_c = bsnpp_pem_cls_loss_func(pred_bm_c[:, 1].contiguous(), gt_iou_map, bm_mask)
    prb_cls_loss_p_c = bsnpp_pem_cls_loss_func(pred_bm_p_c[:, 1].contiguous(), gt_iou_map, bm_mask)
    prb_cls_loss = prb_cls_loss_p + prb_cls_loss_c + prb_cls_loss_p_c

    prb_loss = prb_weight_forward * (prb_reg_loss + prb_cls_loss)

    loss = cbg_loss + prb_loss
    return loss, cbg_loss, prb_loss, cbg_loss_forward, cbg_loss_backward, prb_reg_loss, prb_cls_loss, inter_feature_loss


if __name__ == '__main__':
    batch_size = 2
    temporal_scale = 100
    feature_forward = torch.randn(batch_size, 128, temporal_scale).cuda()
    feature_backward = torch.randn(batch_size, 128, temporal_scale).cuda()
    print(feature_forward.shape, feature_backward.shape)
    inter_feature_loss = cbg_feature_loss(feature_forward, feature_backward)
    print("inter_feature_loss: {}, {}".format(inter_feature_loss.shape, inter_feature_loss))

    bm_mask = get_mask(temporal_scale)
    pred_bm = torch.rand(batch_size, 2, temporal_scale, temporal_scale).cuda()
    pred_start = torch.rand(batch_size, temporal_scale).cuda()
    pred_end = torch.rand(batch_size, temporal_scale).cuda()

    gt_iou_map = torch.rand(batch_size, temporal_scale, temporal_scale)

    pred_start_backward = torch.rand(batch_size, temporal_scale).cuda()
    pred_end_backward = torch.rand(batch_size, temporal_scale).cuda()

    gt_start = torch.rand(batch_size, temporal_scale).cuda()
    gt_end = torch.rand(batch_size, temporal_scale).cuda()

    loss = bsnpp_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask.cuda(),
                           pred_start_backward, pred_end_backward, feature_forward, feature_backward,
                           pred_bm, use_backward_map=True)
    print(loss)

