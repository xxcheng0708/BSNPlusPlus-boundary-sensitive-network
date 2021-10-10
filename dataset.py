# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
from utils import ioa_with_anchors, iou_with_anchors
import os


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self._getDatasetDict()
        # 计算每个r(tn)位置的region区域，r(tn) = [tn−df/2, tn+df/2], where df=tn−tn−1 is the temporal interval between two locations.
        # anchor_xmin和anchor_xmax对应位置组成的（xmin, xmax）就是一个r(tn)的region,后续使用（xmin, xmax）与ts/te热region计算IoR得到TEM的训练数据
        # [-0.05, ..., 0.985]
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        # [0.05, ..., 0.995]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]

    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def __getitem__(self, index):
        # video_data = self._load_file(index)
        video_data = self._load_npy_file(index)
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(
                index, self.anchor_xmin, self.anchor_xmax)
            return video_data, confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _load_file(self, index):
        video_name = self.video_list[index]
        video_df = pd.read_csv(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video_name + ".csv")
        video_data = video_df.values[:, :]
        video_data = torch.Tensor(video_data)  # T * C
        video_data = torch.transpose(video_data, 0, 1)  # C * T
        video_data.float()
        return video_data

    def _load_npy_file(self, index):
        video_name = self.video_list[index]
        # video_npy = pd.read_csv(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video_name + ".csv")
        video_data = np.load(os.path.join(self.feature_path, "anet-bmn-feat-100", "fix_feat_100", video_name + ".npy"))
        # video_data = video_df.values[:, :]
        video_data = torch.Tensor(video_data)  # T * C
        video_data = torch.transpose(video_data, 0, 1)  # C * T
        video_data.float()
        return video_data

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        # 计算segment片段的时长
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            # 把start和end归一化到0-1之间
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        ####################################################################################################
        # generate R_s and R_e，计算GT ts/te的region,前后各延长1.5个temporal_gap
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################
        # 计算BM特征图上每个候选proposal与GT的最大IoU，作为PEM的训练数据
        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        gt_iou_map = torch.Tensor(gt_iou_map)

        ##########################################################################################################
        # calculate the ioa for all timestamp
        # 计算每个r(tn)的region与所有GT ts的最大IoR作为TEM的训练数据
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))

        # 计算每个r(tn)的region与所有GT te的最大IoR作为TEM的训练数据
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=2, shuffle=True,
                                               num_workers=0, pin_memory=True)
    for data, map_score, start_score, end_score in train_loader:
        print(data.shape, map_score.shape, start_score.shape, end_score.shape)
        break

    opt["mode"] = "validate"
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=2, shuffle=False,
                                              num_workers=0, pin_memory=True)
    for i, data in test_loader:
        print(data.shape)
        break
