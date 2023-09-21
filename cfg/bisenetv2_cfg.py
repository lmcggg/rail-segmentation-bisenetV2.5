#!/usr/bin/env python


__all__ = ["BiSeNetV2Config"]


class BiSeNetV2Config:
    img_height = 512
    img_width = 1024
    num_classes = 20
    batch_size = 10
    batch_multiplier = 5
    ohem_ce_loss_thresh = 0.7
    num_epochs =250
    len_epoch = None
    lr_rate = 0.025
    momentum = 0.9
    weight_decay = 5e-4
    burn_in = 1000
    gamma = 0.1
    num_workers = 3
    random_seed = 4
    save_period = 10
    print_after_batch_num = 50
    dataset_name_base = "bisenetv2"
