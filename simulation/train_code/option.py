import argparse
import os


parser = argparse.ArgumentParser(description="PSST-Net simulation training")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default="0")

# Dataset specifications
parser.add_argument("--data_root", type=str, default="../../datasets")
parser.add_argument("--training_set", type=str, default="cave_1024_28")
parser.add_argument("--test_set", type=str, default="TSA_simu_data/Truth")
parser.add_argument("--mask_set", type=str, default="TSA_simu_data")

# Saving specifications
parser.add_argument("--outf", type=str, default="./exp/psst_net/")

# Measurement specifications
parser.add_argument("--input_setting", type=str, default="H")
parser.add_argument("--input_mask", type=str, default="Mask")

# Training specifications
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_epoch", type=int, default=300)
parser.add_argument("--loss_type", type=str, default="L2")
parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
parser.add_argument("--milestones", type=int, nargs="+", default=[50, 100, 150, 200, 250])
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--epoch_sam_num", type=int, default=5000)
parser.add_argument("--learning_rate", type=float, default=4e-4)
parser.add_argument("--RESUME", action="store_true")
parser.add_argument("--re_path", type=str, nargs=2, default=["", ""],
                    metavar=("MODEL_DIR", "RESULT_DIR"))

opt = parser.parse_args()

opt.data_path = os.path.join(opt.data_root, opt.training_set) + os.sep
opt.mask_path = os.path.join(opt.data_root, opt.mask_set)
opt.test_path = os.path.join(opt.data_root, opt.test_set) + os.sep
