"""Evaluate PSST-Net on the ten simulated CASSI test scenes."""

import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from architecture.PSST_Net import SSLT
from utils import LoadTest, init_mask, init_meas, torch_psnr, torch_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="PSST-Net simulation testing")
    parser.add_argument("--data-root", default="../../datasets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="./exp/psst_net/")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--input-setting", default="H")
    parser.add_argument("--input-mask", default="Mask")
    return parser.parse_args()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        checkpoint = checkpoint["net"]
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    data_root = Path(args.data_root)
    truth_dir = str(data_root / "TSA_simu_data" / "Truth") + os.sep
    mask_dir = str(data_root / "TSA_simu_data")

    test_gt = LoadTest(truth_dir).cuda().float()
    mask3d_batch, input_mask = init_mask(mask_dir, args.input_mask, test_gt.shape[0])
    input_meas = init_meas(test_gt, mask3d_batch, args.input_setting)

    model = SSLT(dim=28, stage=2, num_blocks=[3, 2, 2],
                 attention_type="full", input_resolution=256).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()

    with torch.no_grad():
        out1, out2, pred = model(input_meas, input_mask, return_stages=True)

    psnr = [float(torch_psnr(pred[i], test_gt[i]).cpu()) for i in range(test_gt.shape[0])]
    ssim = [float(torch_ssim(pred[i], test_gt[i]).cpu()) for i in range(test_gt.shape[0])]
    prediction = np.transpose(pred.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Test_result.mat"
    sio.savemat(output_path, {"truth": truth, "pred": prediction,
                              "psnr_list": psnr, "ssim_list": ssim})
    print("PSNR per scene:", [round(value, 4) for value in psnr])
    print("SSIM per scene:", [round(value, 4) for value in ssim])
    print("Average PSNR: {:.4f} dB".format(float(np.mean(psnr))))
    print("Average SSIM: {:.4f}".format(float(np.mean(ssim))))
    print("Saved reconstructed HSIs to {}".format(output_path))


if __name__ == "__main__":
    main()
