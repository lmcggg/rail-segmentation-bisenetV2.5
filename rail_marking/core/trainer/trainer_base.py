#!/usr/bin/env python
import logging
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import abc


__all__ = ["TrainerBase"]


class TrainerBase:
    def __init__(
        self,
        model,
        criterion,
        metric_func,
        optimizer,
        data_loaders_dict,
        config,
        scheduler=None,
        device=None,
        logger=None,
    ):
        self._model = model
        self._criterion = criterion
        self._metric_func = metric_func
        self._optimizer = optimizer

        self._logger = logger

        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        self._model = self._model.to(self._device)

        self._start_epoch = 1

        self._train_data_loader = data_loaders_dict["train"]
        self._val_data_loader = data_loaders_dict["val"]
        self._num_train_imgs = len(self._train_data_loader.dataset)
        self._num_val_imgs = len(self._val_data_loader.dataset)

        self._config = config
        self._batch_multiplier = self._config.batch_multiplier
        self._checkpoint_dir = self._config.saved_model_path
        self._num_epochs = self._config.num_epochs
        self._save_period = self._config.save_period
        self._dataset_name_base = self._config.dataset_name_base
        self._print_after_batch_num = self._config.print_after_batch_num

        if self._config.len_epoch is None:
            self._len_epoch = len(self._train_data_loader)
        else:
            self.train_data_loader = TrainerBase.inf_loop(self._train_data_loader)
            self._len_epoch = self._config.len_epoch
        self._do_validation = self._val_data_loader is not None
        self._scheduler = scheduler

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def _train_epoch(self, epoch):

        raise NotImplementedError

    def _calculate_miou(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
        miou = np.nanmean(iou)
        return miou
    def miou(self,data_loader):
        num_classes = 20
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self._device), target.to(self._device)
                output = self._model(data)
                y_true = target.cpu().numpy().ravel()
                y_pred = output.argmax(dim=1).cpu().numpy().ravel()
                confusion_matrix += np.bincount(num_classes * y_true + y_pred, minlength=num_classes ** 2).reshape(
                    num_classes, num_classes)

        intersection = np.diag(confusion_matrix)
        ground_truth_set = confusion_matrix.sum(axis=1)
        predicted_set = confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        iou = intersection / union.astype(np.float32)
        miou = np.mean(iou)
        return miou


    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self._model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        output_file = "checkpoint_{}_epoch_{}.pth".format(arch, epoch)
        if self._dataset_name_base and isinstance(self._dataset_name_base, str) and self._dataset_name_base != "":
            output_file = "{}_{}".format(self._dataset_name_base, output_file)

        filename = os.path.join(self._checkpoint_dir, output_file)
        torch.save(state, filename)

    def resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)

        checkpoint = torch.load(resume_path)
        self._start_epoch = checkpoint["epoch"] + 1

        self._model.load_state_dict(checkpoint["state_dict"])

        self._optimizer.load_state_dict(checkpoint["optimizer"])

    def train(self):
        logging.info("========================================")
        logging.info("Start training {}".format(type(self._model).__name__))
        logging.info("========================================")
        logs = []
        train_loss_list = []
        val_loss_list = []
        miou_list = []

        for epoch in range(self._start_epoch, self._num_epochs + 1):
            import pandas as pd
            import requests
            import pandas
            import openpyxl


            train_loss, val_loss = self._train_epoch(epoch)
            miou=self.miou(self._val_data_loader)
            miou_list.append(miou)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            # create a dataframe for train loss and save it to excel
            train_loss_df = pd.DataFrame({'train_loss': train_loss_list})
            train_loss_df.to_excel('train_loss.xlsx', index=False)

            # create a dataframe for val loss and save it to excel
            val_loss_df = pd.DataFrame({'val_loss': val_loss_list})
            val_loss_df.to_excel('val_loss.xlsx', index=False)
            miou_df = pd.DataFrame({'miou': miou_list})
            miou_df.to_excel('miou.xlsx', index=False)


            log_epoch = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,

            }

            logging.info("========================================")
            logging.info("epoch: {}, train_loss: {: .4f}, val_loss: {: .4f}".format(epoch, train_loss, val_loss))
            logging.info("========================================")
            logs.append(log_epoch)
            if self._logger:
                self._logger.add_scalar("train/train_loss", train_loss, epoch)
                self._logger.add_scalar("val/val_loss", val_loss, epoch)

            if (epoch + 1) % self._save_period == 0:
                self._save_checkpoint(epoch, save_best=True)

        return logs

    @staticmethod
    def inf_loop(data_loader):
        from itertools import repeat

        for loader in repeat(data_loader):
            yield from loader
