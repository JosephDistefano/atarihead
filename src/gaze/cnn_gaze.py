import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from pathlib import Path
import yaml
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
from data_processing.data_utils import ImbalancedDatasetSampler
from data_processing.data_loaders import HDF5TorchDataset, load_data_iter


class CNN_GAZE(nn.Module):
    def __init__(
        self,
        input_shape=(84, 84),
        load_model=False,
        epoch=0,
        num_actions=18,
        game="breakout",
        data_types=["images", "gazes"],
        dataset_train="combined",
        dataset_train_load_type="disk",
        dataset_val="combined",
        dataset_val_load_type="disk",
        device=torch.device("cpu"),
        mode="train",
    ):
        super(CNN_GAZE, self).__init__()
        self.game = game
        self.data_types = data_types
        self.input_shape = input_shape
        self.num_actions = num_actions

        config_path = Path(__file__).parents[1] / "src/config.yaml"
        self.config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

        model_save_dir = os.path.join(
            self.config["model_save_dir_gaze"], game, dataset_train
        )
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.model_save_string = os.path.join(
            model_save_dir, self.__class__.__name__ + "_Epoch_{}.pt"
        )
        log_dir = os.path.join(
            self.config_yml["RUNS_DIR"],
            game,
            "{}_{}".format(dataset_train, self.__class__.__name__),
        )
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                log_dir,
                "run_{}".format(
                    len(os.listdir(log_dir)) if os.path.exists(log_dir) else 0
                ),
            )
        )
        self.device = device
        self.batch_size = self.config_yml["BATCH_SIZE"]
        if mode != "eval":
            self.train_data_iter = load_data_iter(
                game=self.game,
                data_types=self.data_types,
                dataset=dataset_train,
                dataset_exclude=dataset_val,
                device=device,
                batch_size=self.batch_size,
                sampler=ImbalancedDatasetSampler,
                load_type=dataset_train_load_type,
            )

            self.val_data_iter = load_data_iter(
                game=self.game,
                data_types=self.data_types,
                dataset=dataset_val,
                device=device,
                batch_size=self.batch_size,
                load_type=dataset_val_load_type,
            )

        self.conv1 = nn.Conv2d(4, 32, 8, stride=(4, 4))
        self.pool = nn.MaxPool2d((1, 1), (1, 1), (0, 0), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(32, 1, 8, stride=(4, 4))

        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.load_model = load_model
        self.epoch = epoch

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.deconv1(x)))
        x = self.pool(F.relu(self.deconv2(x)))
        x = self.deconv3(x)
        x = x.squeeze(1)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = F.log_softmax(x, dim=1)
        return x

    def out_shape(self, layer, in_shape):
        h_in, w_in = in_shape
        h_out, w_out = floor(
            (
                (
                    h_in
                    + 2 * layer.padding[0]
                    - layer.dilation[0] * (layer.kernel_size[0] - 1)
                    - 1
                )
                / layer.stride[0]
            )
            + 1
        ), floor(
            (
                (
                    w_in
                    + 2 * layer.padding[1]
                    - layer.dilation[1] * (layer.kernel_size[1] - 1)
                    - 1
                )
                / layer.stride[1]
            )
            + 1
        )
        return h_out, w_out

    def lin_in_shape(self):
        out_shape = self.out_shape(self.conv1, self.input_shape)
        out_shape = self.out_shape(self.conv2, out_shape)
        out_shape = self.out_shape(self.conv3, out_shape)
        return out_shape

    def loss_fn(self, loss_, smax_pi, targets):
        targets_reshpaed = targets.view(-1, targets.shape[1] * targets.shape[2])

        kl_loss = loss_(smax_pi, targets_reshpaed)

        return kl_loss

    def train_loop(self, opt, lr_scheduler, loss_, batch_size=32, gaze_pred=None):
        self.loss_ = loss_
        if self.load_model:
            model_pickle = torch.load(self.model_save_string.format(self.epoch))
            self.load_state_dict(model_pickle["model_state_dict"])
            opt.load_state_dict(model_pickle["model_state_dict"])
            self.epoch = model_pickle["epoch"]
            loss_val = model_pickle["loss"]
        eix = 0
        for epoch in range(self.epoch, 15):
            for i, data in enumerate(self.train_data_iter):
                x, y = self.get_data(data)

                opt.zero_grad()

                smax_pi = self.forward(x)

                loss = self.loss_fn(loss_, smax_pi, y)
                loss.backward()
                opt.step()
                # self.writer.add_scalar('Loss', loss.data.item(),
                #                        (epoch + 1) * i)
                self.writer.add_scalar("Loss", loss.data.item(), eix)
                eix += 1

            if epoch % 1 == 0:
                # self.writer.add_histogram('smax', smax_pi[0])
                # self.writer.add_histogram('target', y)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "loss": loss,
                    },
                    self.model_save_string.format(epoch),
                )
            print("Epoch ", epoch, "loss", loss)
            self.writer.add_scalar("Epoch Loss", loss.data.item(), epoch)
            self.writer.add_scalar("Learning Rate", lr_scheduler.get_lr()[0], epoch)
            # self.writer.add_scalar('Epoch Val Loss',
            #                        self.val_loss().data.item(), epoch)
            if epoch % 2 == 0:
                lr_scheduler.step()

    def get_data(self, data):
        if isinstance(data, dict):
            x = data["images"]
            y = data["gazes"]

        elif isinstance(data, list):
            x, y = data

        return x, y

    def val_loss(self):
        self.eval()
        val_loss = []
        with torch.no_grad():
            for i, data in enumerate(self.val_data_iter):
                x, y = self.get_data(data)
                smax_pi = self.forward(x)

                val_loss.append(self.loss_fn(self.loss_, smax_pi, y))
        self.train()
        return torch.mean(torch.Tensor(val_loss))

    def infer(self, x_var):
        self.eval()

        with torch.no_grad():
            smax_dist = self.forward(x_var).view(
                -1, self.input_shape[0], self.input_shape[1]
            )

        self.train()

        return smax_dist

    def load_model_fn(self, epoch):
        self.epoch = epoch
        model_pickle = torch.load(self.model_save_string.format(self.epoch))
        self.load_state_dict(model_pickle["model_state_dict"])

        self.epoch = model_pickle["epoch"]
        loss_val = model_pickle["loss"]
        print(
            "Loaded {} model from saved checkpoint {} with loss {}".format(
                self.__class__.__name__, self.epoch, loss_val
            )
        )
