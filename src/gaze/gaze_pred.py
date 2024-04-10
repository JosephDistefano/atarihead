import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from gaze.cnn_gaze import CNN_GAZE
from data_processing.data_loaders import load_action_data, load_gaze_data

from data_processing.data_loaders import load_hdf_data
import numpy as np
from feat_utils import gaze_pdf
import argparse

# pylint: disable=all
from feat_utils import (
    image_transforms,
    reduce_gaze_stack,
    draw_figs,
    fuse_gazes,
    fuse_gazes_noop,
)  # nopep8
from torch.optim.lr_scheduler import LambdaLR


def gaze_prediction(config, game, MODE="train"):

    BATCH_SIZE = config["BATCH_SIZE"]

    dataset_train = dataset_val = "combined"

    device = torch.device("cuda")

    data_types = ["images", "gazes"]

    gaze_net = CNN_GAZE(
        game=game,
        data_types=data_types,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_train_load_type="chunked",
        dataset_val_load_type="chunked",
        device=device,
        mode="train",
    ).to(device=device)

    ######
    optimizer = torch.optim.Adadelta(gaze_net.parameters(), lr=5e-1, rho=0.95)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
    # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda e:0.8)
    # optimizer = torch.optim.SGD(gaze_net.parameters(), lr=0.1)

    # Define your lambda function for multiplicative decay
    lambda_func = lambda epoch: 0.95**epoch

    # Create the scheduler
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_func)
    # lr_scheduler = None
    loss_ = torch.nn.KLDivLoss(reduction="batchmean")

    if MODE == "eval":
        curr_group_data = load_hdf_data(
            game=game,
            dataset=dataset_val,
            data_types=["images", "gazes"],
        )

        x, y = curr_group_data.values()
        x = x[0]
        y = y[0]

        image_ = x[34]
        gaze_ = y[34]

        for cpt in tqdm(range(14, 15, 1)):
            gaze_net.epoch = cpt
            gaze_net.load_model_fn(cpt)
            smax = (
                gaze_net.infer(torch.Tensor(image_).to(device=device).unsqueeze(0))
                .squeeze()
                .cpu()
                .data.numpy()
            )

            # gaze_max = np.array(gaze_max)/84.0
            # smax = gaze_pdf([g_max])
            # gaze_ = gaze_pdf([gaze_max])
            pile = np.percentile(smax, 90)
            smax = np.clip(smax, pile, 1)
            smax = (smax - np.min(smax)) / (np.max(smax) - np.min(smax))
            smax = smax / np.sum(smax)

            draw_figs(x_var=smax, gazes=gaze_ * 255)
            draw_figs(x_var=image_[-1], gazes=gaze_ * 255)

    else:
        gaze_net.train_loop(optimizer, lr_scheduler, loss_, batch_size=BATCH_SIZE)
    return

    # if __name__ == "__main__":
    # rand_image = torch.rand(4, 84, 84)
    # rand_target = torch.rand(4, 84, 84)

    # cnn_gaze_net = CNN_GAZE()
    # cnn_gaze_net.lin_in_shape()
    # optimizer = torch.optim.Adadelta(cnn_gaze_net.parameters(), lr=1.0, rho=0.95)

    # # if scheduler is declared, ought to use & update it , else model never trains
    # # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    # #     optimizer, lr_lambda=lambda x: x*0.95)
    # lr_scheduler = None

    # loss_ = torch.nn.KLDivLoss(reduction="batchmean")
    # cnn_gaze_net.train_loop(
    #     optimizer, lr_scheduler, loss_, rand_image, rand_target, batch_size=4
    # )
