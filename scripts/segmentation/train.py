#!/usr/bin/env python
#pip install -U albumentations[imgaug]
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as abm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.models import BiSeNetV2, OHEMCELoss
    from rail_marking.segmentation.data_loader import Rs19dDataset, DataTransformBase
    from rail_marking.segmentation.trainer import BiSeNetV2Trainer
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


def train_process(data_path, config):
    def _worker_init_fn_():
        import random
        import numpy as np
        import torch

        random_seed = config.random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

    input_size = (config.img_height, config.img_width)

    transforms = [
        abm.RandomResizedCrop(
            scale=(0.25, 2),
            height=config.img_height,
            width=config.img_width,
            always_apply=True,
        ),
        abm.OneOf([abm.IAAAdditiveGaussianNoise(), abm.GaussNoise()], p=0.5),
        abm.OneOf(
            [
                abm.MedianBlur(blur_limit=3),
                abm.GaussianBlur(blur_limit=3),
                abm.MotionBlur(blur_limit=3),
            ],
            p=0.1,
        ),
        abm.RandomGamma(gamma_limit=(80, 120), p=0.5),
        abm.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.5),
        abm.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        abm.RandomShadow(p=0.5),
        abm.ChannelShuffle(p=0.5),
        abm.HorizontalFlip(p=0.5),
        abm.Cutout(num_holes=100, max_w_size=8, max_h_size=8, p=0.5),
        abm.Rotate(limit=10, p=0.5, border_mode=0),
    ]

    data_transform = DataTransformBase(transforms=transforms, input_size=input_size, normalize=True)
    train_dataset = Rs19dDataset(data_path=data_path, phase="train", transform=data_transform)
    val_dataset = Rs19dDataset(data_path=data_path, phase="val", transform=data_transform)


    weighted_values =[13.41229074,23.58617656 ,7.81774638 ,35.82407711, 25.22984806 ,20.71561984,
 47.01028036, 47.03716904,  4.65097763, 11.65627556,  4.77515867, 47.08544603,
 11.28928189, 42.44120373, 49.26380146,  7.81770225, 30.11820687, 21.45696974,
 44.21305235 ,50.49834979]#可以自己计算，这里是对我自己的训练图像计算得到的结果
    # train_dataset.weighted_class()


    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        worker_init_fn=_worker_init_fn_(),
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
    )
    data_loaders_dict = {"train": train_data_loader, "val": val_data_loader}
    model = BiSeNetV2(n_classes=config.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = OHEMCELoss(thresh=config.ohem_ce_loss_thresh, weighted_values=weighted_values)

    base_lr_rate = config.lr_rate / (config.batch_size * config.batch_multiplier)
    base_weight_decay = config.weight_decay * (config.batch_size * config.batch_multiplier)

    def _lambda_epoch(epoch):
        import math

        max_epoch = config.num_epochs
        return math.pow((1 - epoch * 1.0 / max_epoch), 0.9)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr_rate,
        momentum=config.momentum,
        weight_decay=base_weight_decay,
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda_epoch)
    trainer = BiSeNetV2Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        data_loaders_dict=data_loaders_dict,
        config=config,
        scheduler=scheduler,
        device=device,
    )

    if config.snapshot and os.path.isfile(config.snapshot):
        trainer.resume_checkpoint(config.snapshot)

    with torch.autograd.set_detect_anomaly(True):
        trainer.train()
import matplotlib.pyplot as plt
def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main(args):
    config = BiSeNetV2Config()
    config.saved_model_path = args.saved_model_path
    config.snapshot = args.snapshot
    train_process(args.data_path, config)
    train_loss, val_loss = train_process(args.data_path, config)

    # Call plot_loss() with the train_loss and val_loss lists
    plot_loss(train_loss, val_loss)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--saved_model_path", type=str, required=True)
    parser.add_argument("--snapshot", type=str, required=False)
    parsed_args = parser.parse_args()

    main(parsed_args)
