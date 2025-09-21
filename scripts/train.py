import sys
import json
import logging

from numpy import require

from VSDataset import VSDataset

import torch
import argparse
from ml_framework.models.UNet import UNet
from ml_framework.framework.SupervisedMLFramework import SupervisedMLFramework
from ml_framework.losses.cross_entropy_loss import CrossEntropyLoss
from ml_framework.transforms.StandardizeTransform import StandardizeTransform
import ml_framework.framework.Metrics as metrics

logging.getLogger("pyvips").setLevel(logging.ERROR)

def main(cfg):

    try:
        data_path = cfg["data_path"]
        mask_path = cfg["mask_path"]
        output_path = cfg["output_path"]
    except Exception as e:
        raise Exception("Failed to read necessary values out of config: {e}")

    train_dataset = VSDataset(data_path, mask_path)
    model = UNet(in_ch=3)

    """ --- Hyper Parameters --- """
    lr = .0001
    epochs = 100
    weight_decay = .0001

    loss_function = CrossEntropyLoss()
    optim = torch.optim.Adam
    optim_params = {"lr": lr, "weight_decay": weight_decay}
    scheduler = torch.optim.lr_scheduler.StepLR
    sched_params={"step_size": 30}
    framework = SupervisedMLFramework(model, "UNet", output_path, train_dataset)
    framework.train(epochs, loss_function, optim, optim_params, sched=scheduler,
                    sched_params=sched_params, batch_size=16, weight_save_period=5,
                    patience=20, validation_percent=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", required=True)

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load config: {e}")

    main(cfg)



