import sys
from dataset import VideoDataSet
from loss_function import bsnpp_loss_func, get_mask
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from models import BSNPP
import pandas as pd
from post_processing import BSNPP_post_processing
from eval import evaluation_proposal
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn

torch.manual_seed(100)
sys.dont_write_bytecode = True


def train_BSNPP(data_loader, model, optimizer, epoch, bm_mask, writer):
    model.train()
    # 总损失
    epoch_loss = 0
    # CBG模块损失
    epoch_cbg_loss = 0
    epoch_cbg_loss_forward = 0
    epoch_cbg_loss_backward = 0
    epoch_cbg_feature_loss = 0
    # PRB模块正向损失
    epoch_prb_loss = 0
    epoch_prb_reg_loss = 0
    epoch_prb_cls_loss = 0

    for n_iter, (input_data, label_confidence, label_start, label_end) in tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader)):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map_p, confidence_map_c, confidence_map_p_c, start_forward, end_forward, feature_forward = model(input_data)

        input_data_backward = torch.flip(input_data, dims=(2,))
        _, _, _, start_backward, end_backward, feature_backward = model(input_data_backward)

        loss = bsnpp_loss_func(confidence_map_p, confidence_map_c, confidence_map_p_c,
                               start_forward, end_forward, label_confidence, label_start, label_end,
                               bm_mask.cuda(), start_backward, end_backward, feature_forward, feature_backward)
        iter_loss, cbg_loss, prb_loss, cbg_loss_forward, cbg_loss_backward, prb_reg_loss, prb_cls_loss, cbg_feature_loss = loss

        optimizer.zero_grad()
        iter_loss.backward()
        optimizer.step()

        print("cbg loss: {:.03f}, prb loss: {:.03f}, iter loss: {:.03f}".format(cbg_loss, prb_loss, iter_loss))
        writer.add_scalar("train iter loss/cbg_loss", cbg_loss.item(), epoch * len(data_loader) + n_iter + 1)
        writer.add_scalar("train iter loss/prb_loss", prb_loss.item(), epoch * len(data_loader) + n_iter + 1)
        writer.add_scalar("train iter loss/cbg_loss_forward", cbg_loss_forward.item(),
                          epoch * len(data_loader) + n_iter + 1)
        writer.add_scalar("train iter loss/cbg_loss_backward", cbg_loss_backward.item(),
                          epoch * len(data_loader) + n_iter + 1)
        writer.add_scalar("train iter loss/prb_reg_loss", prb_reg_loss.item(), epoch * len(data_loader) + n_iter + 1)
        writer.add_scalar("train iter loss/prb_cls_loss", prb_cls_loss.item(), epoch * len(data_loader) + n_iter + 1)
        writer.add_scalar("train iter loss/cbg_feature_loss", cbg_feature_loss.item(),
                          epoch * len(data_loader) + n_iter + 1)
        writer.add_scalar("train iter loss/iter_loss", iter_loss.item(), epoch * len(data_loader) + n_iter + 1)

        epoch_cbg_loss += cbg_loss.cpu().detach().numpy()
        epoch_prb_loss += prb_loss.cpu().detach().numpy()
        epoch_loss += iter_loss.cpu().detach().numpy()
        epoch_cbg_loss_forward += cbg_loss_forward.cpu().detach().numpy()
        epoch_cbg_loss_backward += cbg_loss_backward.cpu().detach().numpy()
        epoch_cbg_feature_loss = cbg_feature_loss.cpu().detach().numpy()
        epoch_prb_reg_loss += prb_reg_loss.cpu().detach().numpy()
        epoch_prb_cls_loss += prb_cls_loss.cpu().detach().numpy()

    print("BSNPP training loss(epoch {}): total_loss: {:.3f}, "
          "cbg_loss: {:.3f}, cbg_loss_forward: {:.3f}, cbg_loss_backward: {:.3f}, cbg_feature_loss: {:.3f}, "
          "prb_loss: {:.3f}, prb_reg_loss: {:.3f}, prb_cls_loss: {:.3f}".format(
        epoch, epoch_loss / (n_iter + 1), epoch_cbg_loss / (n_iter + 1), epoch_cbg_loss_forward / (n_iter + 1),
               epoch_cbg_loss_backward / (n_iter + 1), epoch_cbg_feature_loss / (n_iter + 1),
               epoch_prb_loss / (n_iter + 1), epoch_prb_reg_loss / (n_iter + 1), epoch_prb_cls_loss / (n_iter + 1)
    ))
    writer.add_scalar("train epoch loss/cbg_loss", epoch_cbg_loss / (n_iter + 1), epoch)
    writer.add_scalar("train epoch loss/prb_loss", epoch_prb_loss / (n_iter + 1), epoch)
    writer.add_scalar("train epoch loss/cbg_loss_forward", epoch_cbg_loss_forward / (n_iter + 1), epoch)
    writer.add_scalar("train epoch loss/cbg_loss_backward", epoch_cbg_loss_backward / (n_iter + 1), epoch)
    writer.add_scalar("train epoch loss/prb_reg_loss", epoch_prb_reg_loss / (n_iter + 1), epoch)
    writer.add_scalar("train epoch loss/prb_cls_loss", epoch_prb_cls_loss / (n_iter + 1), epoch)
    writer.add_scalar("train epoch loss/cbg_feature_loss", epoch_cbg_feature_loss / (n_iter + 1), epoch)
    writer.add_scalar("train epoch loss/total_loss", epoch_loss / (n_iter + 1), epoch)


def test_BSNPP(data_loader, model, epoch, bm_mask, writer, best_loss):
    model.eval()
    # 总损失
    epoch_loss = 0
    # CBG模块损失
    epoch_cbg_loss = 0
    epoch_cbg_loss_forward = 0
    epoch_cbg_loss_backward = 0
    epoch_cbg_feature_loss = 0
    # PRB模块正向损失
    epoch_prb_loss = 0
    epoch_prb_reg_loss = 0
    epoch_prb_cls_loss = 0

    for n_iter, (input_data, label_confidence, label_start, label_end) in tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader)):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map_p, confidence_map_c, confidence_map_p_c, start_forward, end_forward, feature_forward = model(input_data)

        input_data_backward = torch.flip(input_data, dims=(2,))
        _, _, _, start_backward, end_backward, feature_backward = model(input_data_backward)

        loss = bsnpp_loss_func(confidence_map_p, confidence_map_c, confidence_map_p_c,
                               start_forward, end_forward, label_confidence, label_start, label_end,
                               bm_mask.cuda(), start_backward, end_backward, feature_forward, feature_backward)

        iter_loss, cbg_loss, prb_loss, cbg_loss_forward, cbg_loss_backward, prb_reg_loss, prb_cls_loss, cbg_feature_loss = loss

        epoch_cbg_loss += cbg_loss.cpu().detach().numpy()
        epoch_prb_loss += prb_loss.cpu().detach().numpy()
        epoch_loss += iter_loss.cpu().detach().numpy()
        epoch_cbg_loss_forward += cbg_loss_forward.cpu().detach().numpy()
        epoch_cbg_loss_backward += cbg_loss_backward.cpu().detach().numpy()
        epoch_cbg_feature_loss = cbg_feature_loss.cpu().detach().numpy()
        epoch_prb_reg_loss += prb_reg_loss.cpu().detach().numpy()
        epoch_prb_cls_loss += prb_cls_loss.cpu().detach().numpy()

    print("BSNPP validate loss(epoch {}): total_loss: {:.3f}, "
          "cbg_loss: {:.3f}, cbg_loss_forward: {:.3f}, cbg_loss_backward: {:.3f}, cbg_feature_loss: {:.3f}, "
          "prb_loss: {:.3f}, prb_reg_loss: {:.3f}, prb_cls_loss: {:.3f}".format(
        epoch, epoch_loss / (n_iter + 1), epoch_cbg_loss / (n_iter + 1), epoch_cbg_loss_forward / (n_iter + 1),
        epoch_cbg_loss_backward / (n_iter + 1), epoch_cbg_feature_loss / (n_iter + 1),
        epoch_prb_loss / (n_iter + 1), epoch_prb_reg_loss / (n_iter + 1), epoch_prb_cls_loss / (n_iter + 1)
    ))
    writer.add_scalar("test epoch loss/cbg_loss", epoch_cbg_loss / (n_iter + 1), epoch)
    writer.add_scalar("test epoch loss/prb_loss", epoch_prb_loss / (n_iter + 1), epoch)
    writer.add_scalar("test epoch loss/cbg_loss_forward", epoch_cbg_loss_forward / (n_iter + 1), epoch)
    writer.add_scalar("test epoch loss/cbg_loss_backward", epoch_cbg_loss_backward / (n_iter + 1), epoch)
    writer.add_scalar("test epoch loss/prb_reg_loss", epoch_prb_reg_loss / (n_iter + 1), epoch)
    writer.add_scalar("test epoch loss/prb_cls_loss", epoch_prb_cls_loss / (n_iter + 1), epoch)
    writer.add_scalar("test epoch loss/cbg_feature_loss", epoch_cbg_feature_loss / (n_iter + 1), epoch)
    writer.add_scalar("test epoch loss/total_loss", epoch_loss / (n_iter + 1), epoch)

    state = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BSNPP_checkpoint_{}_{:.4f}.pth.tar".format(epoch, epoch_loss / (n_iter + 1)))
    if epoch_loss / (n_iter + 1) < best_loss:
        best_loss = epoch_loss / (n_iter + 1)
        torch.save(state, opt["checkpoint_path"] + "/BSNPP_best.pth.tar")
    return best_loss


def BSNPP_Train(opt, writer):
    model = BSNPP(opt)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])

    # bm_mask是上三角矩阵，右上方全1，左下方全0，因为BM层有一半的proposal起始位置已经超过了视频总长度，是无效的
    bm_mask = get_mask(opt["temporal_scale"])
    bm_mask = nn.Parameter(bm_mask, requires_grad=False)

    best_loss = 1e10
    for epoch in range(opt["train_epochs"]):
        scheduler.step()
        train_BSNPP(train_loader, model, optimizer, epoch, bm_mask, writer)
        best_loss = test_BSNPP(test_loader, model, epoch, bm_mask, writer, best_loss)


def BSNPP_inference(opt):
    model = BSNPP(opt)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/BSNPP_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in tqdm.tqdm(test_loader, total=len(test_loader)):
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map_p, confidence_map_c, confidence_map_p_c, start_forward, end_forward, _ = model(input_data)
            confidence_map = (confidence_map_p + confidence_map_c + confidence_map_p_c) / 3

            input_data_backward = torch.flip(input_data, dims=(2,))
            _, _, _, start_backward, end_backward, _ = model(input_data_backward)

            start = torch.sqrt(start_forward * torch.flip(end_backward, dims=(1,)))
            end = torch.sqrt(end_forward * torch.flip(start_backward, dims=(1,)))

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and end_index < tscale:
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BSNPP_results/" + video_name + ".csv", index=False)


def main(opt):
    if opt["mode"] == "train":
        writer = SummaryWriter(opt["checkpoint_path"])
        print(writer)
        BSNPP_Train(opt, writer)
        writer.close()
    elif opt["mode"] == "inference":
        if not os.path.exists("output/BSNPP_results"):
            os.makedirs("output/BSNPP_results")
        BSNPP_inference(opt)
        print("Post processing start")
        BSNPP_post_processing(opt)
        print("Post processing finished")
        evaluation_proposal(opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    main(opt)
