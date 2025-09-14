import sys
sys.path.append('..')

import torch
from datasets.OEDDataset import OEDDataset
from models.InceptionBinary import InceptionBinary
from framework.SupervisedMLFramework import SupervisedMLFramework
from losses.cross_entropy_loss import CrossEntropyLoss
from transforms.StandardizeTransform import StandardizeTransform
from models.VGGBinary import VGGBinary
import framework.Metrics as metrics

DATA_PATH = "/projectsp/foran/yl1685/data/patches_lev0"
OUTPUT_PATH = "../../output/train_output"

INPUT_SIZE = 512
USE_CUSTOM_VALIDATION = True

if USE_CUSTOM_VALIDATION:
    validation_dataset = OEDDataset(DATA_PATH, "validation", output_path=OUTPUT_PATH)

train_dataset = OEDDataset(DATA_PATH, "train", output_path=OUTPUT_PATH, patch_subset_percent=.5)

model = VGGBinary(pretrained=False)

""" --- Hyper Parameters --- """
lr = .0001
epochs = 100
weight_decay = .001

weights = torch.tensor([1, 1], dtype=torch.float32).to("cuda")
loss_function = CrossEntropyLoss(weight=weights)
optim = torch.optim.Adam
optim_params = {"lr": lr}
scheduler = torch.optim.lr_scheduler.StepLR
sched_params={"step_size": 30}
framework = SupervisedMLFramework(model, "VGGBinary", OUTPUT_PATH, train_dataset, custom_validation_dataset= validation_dataset if USE_CUSTOM_VALIDATION else None)
framework.train(epochs, loss_function, optim, optim_params, sched=scheduler, save_metric=metrics.calc_weighted_f1_score, save_metric_direction= ">", sched_params=sched_params, batch_size=16, weight_save_period=5, patience=100, use_custom_validation_set=USE_CUSTOM_VALIDATION, validation_percent=20)
