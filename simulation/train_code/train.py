import os
import time
import datetime
import numpy as np
import scipy.io as scio
from tqdm import tqdm

from option import opt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
from torch.autograd import Variable

from utils import *
from architecture.PSST_Net import SSLT as SSLT93

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception("NO GPU!")


date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + str("0.1_0.3_1.0") + "/" + date_time + "SSLT93/result/"
model_path = opt.outf + str("0.1_0.3_1.0") + "/" + date_time + "SSLT93/model/"

if opt.RESUME:
    model_path = opt.re_path[0]
    result_path = opt.re_path[1]

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

logger = gen_log(model_path)
logger.info("\n trainSetting:{}\n".format(opt))
logger.info("torch.cuda.device_count() = {}".format(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    logger.info("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))


def _unwrap_model(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m


model = SSLT93(dim=28, stage=2, num_blocks=[3, 2, 2], attention_type="full", input_resolution=256).cuda()
gpu_ids = [x.strip() for x in str(opt.gpu_id).split(",") if x.strip() != ""]
if len(gpu_ids) >= 2 and torch.cuda.device_count() >= 2:
    model = torch.nn.DataParallel(model)
    logger.info("Using DataParallel (visible GPUs={})".format(torch.cuda.device_count()))
else:
    logger.info("Using single GPU")


mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)

train_set = LoadTraining(opt.data_path)
test_data = LoadTest(opt.test_path)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler == "MultiStepLR":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
else:
    raise ValueError("Unknown scheduler: {}".format(opt.scheduler))

if opt.loss_type == "L2":
    mse = torch.nn.MSELoss().cuda()
else:
    raise ValueError("Unknown loss_type: {}".format(opt.loss_type))

DEEP_SUPERVISION_W = (0.1, 0.3, 1.0)


def _forward_three_stage(model_obj, input_meas, input_mask):
    out1, out2, out3 = model_obj(input_meas, input_mask, return_stages=True)
    return out1, out2, out3


def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    train_logger = tqdm(range(batch_num))
    for i in train_logger:
        gt_batch = shuffle_crop(train_set, opt.batch_size)
        gt = Variable(gt_batch).cuda().float()
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)

        optimizer.zero_grad()
        out1, out2, out3 = _forward_three_stage(model, input_meas, input_mask_train)

        loss1 = torch.sqrt(mse(out1, gt))
        loss2 = torch.sqrt(mse(out2, gt))
        loss3 = torch.sqrt(mse(out3, gt))
        w1, w2, w3 = DEEP_SUPERVISION_W
        loss = w1 * loss1 + w2 * loss2 + w3 * loss3

        epoch_loss += loss.data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_logger.set_description(
            desc="[epoch: %d][lr: %.6f][loss: %.6f][mean_loss: %.6f]"
            % (epoch, scheduler.get_last_lr()[0], loss, epoch_loss / (i + 1))
        )
    end = time.time()
    logger.info(
        "===> Epoch {} Complete: lr:{:.6f} Avg. Loss: {:.6f} time: {:.2f}".format(
            epoch, scheduler.get_last_lr()[0], epoch_loss / batch_num, (end - begin)
        )
    )
    return 0


def test(epoch, logger):
    psnr_list, ssim_list = [], []
    psnr_s1, psnr_s2 = [], []
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)

    model.eval()
    begin = time.time()
    with torch.no_grad():
        out1, out2, out3 = _forward_three_stage(model, input_meas, input_mask_test)
    end = time.time()

    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(out3[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(out3[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        psnr_s1.append(torch_psnr(out1[k], test_gt[k]).detach().cpu().numpy())
        psnr_s2.append(torch_psnr(out2[k], test_gt[k]).detach().cpu().numpy())

    pred = np.transpose(out3.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = float(np.mean(np.asarray(psnr_list)))
    ssim_mean = float(np.mean(np.asarray(ssim_list)))
    psnr_mean_s1 = float(np.mean(np.asarray(psnr_s1)))
    psnr_mean_s2 = float(np.mean(np.asarray(psnr_s2)))

    logger.info(
        "===> Epoch {}: testing psnr(s1/s2/s3) = {:.2f}/{:.2f}/{:.2f}, ssim = {:.3f}, time: {:.2f}".format(
            epoch, psnr_mean_s1, psnr_mean_s2, psnr_mean, ssim_mean, (end - begin)
        )
    )
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean


def main():
    psnr_max = 0
    start_epoch = 0

    if opt.RESUME:
        path_checkpoint = os.path.join(model_path, "mycheckpoint.pth")
        recheckpoint = torch.load(path_checkpoint, map_location="cuda")
        _unwrap_model(model).load_state_dict(recheckpoint["net"])
        optimizer.load_state_dict(recheckpoint["optimizer"])
        scheduler.load_state_dict(recheckpoint["scheduler"])
        start_epoch = recheckpoint["epoch"] + 1
        psnr_max = recheckpoint["psnr_max"]
        logger.info("Resume from epoch {} with psnr_max {:.2f}".format(start_epoch, psnr_max))

    for epoch in range(start_epoch + 1, opt.max_epoch + 1):
        train(epoch, logger)
        pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean = test(epoch, logger)
        scheduler.step()

        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            name = "Test_result_epoch_{}_{}.mat".format(epoch, psnr_mean)
            scio.savemat(os.path.join(result_path, name), {"truth": truth, "pred": pred})
            checkpoint(_unwrap_model(model), epoch, model_path, logger)

        torch.save(
            {
                "epoch": epoch,
                "net": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "psnr_max": psnr_max,
            },
            os.path.join(model_path, "mycheckpoint.pth"),
        )


if __name__ == "__main__":
    main()
