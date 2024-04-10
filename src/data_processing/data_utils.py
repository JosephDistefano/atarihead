import os
import csv
import pandas as pd
import numpy as np

from collections import Counter
import torch
from torchvision import transforms
from scipy.stats import multivariate_normal


def get_game_entries_(game_dir):
    game_dir_entries = os.listdir(game_dir)
    game_runs = []
    game_runs_dirs = []
    game_runs_gaze = []
    for entry in game_dir_entries:
        if entry.__contains__(".txt"):
            game_runs.append(entry.split(".txt")[0])
            game_runs_dirs.append(game_dir)
            game_runs_gaze.append(entry)
    return game_runs, game_runs_dirs, game_runs_gaze


def process_gaze_data(gaze_read_file, gaze_save_file, valid_actions):
    game_run_data = []
    # get gaze data from text file and put it into list

    with open(gaze_read_file, "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            game_run_data.append(row)

    header = game_run_data[0]
    game_run_data = game_run_data[1:]
    game_run_data_mod = []

    # Loop through data and make it [x,y] instead of a long list
    for tstep in game_run_data:
        other_data_without_gaze, gaze_data = [], []
        other_data_without_gaze = tstep[: len(header) - 1]
        gaze_data = tstep[len(header) - 1 :]
        # If there is missing data it copies the previous data point before it
        if len(gaze_data) == 1 and gaze_data[0] == "null":
            gaze_data = game_run_data_mod[-1][len(header) - 1]
            gaze_data_ = gaze_data
        else:
            gaze_data_ = [
                [float(gd) for gd in gaze_data[ix : ix + 2]]
                for ix in range(0, len(gaze_data) - 1, 2)
            ]
        other_data_without_gaze.append(gaze_data_)
        game_run_data_mod.append(other_data_without_gaze)

    game_run_data_mod_df = pd.DataFrame(game_run_data_mod, columns=header)

    # changes the actions to integers and checks it is not null and in valid actions
    game_run_data_mod_df["action"] = game_run_data_mod_df["action"].apply(
        lambda x: 0 if x == "null" else (x if int(x) in valid_actions else 0)
    )
    game_run_data_mod_df.to_csv(gaze_save_file)


def stack_data(images, targets, targets_2, stack=1, stack_type="", stacking_skip=1):
    if stack > 0:
        images_ = []
        targets_ = []
        targets2_ = []
        for ix in range(0, len(targets) - stack, stacking_skip):
            images_.append(images[ix : ix + stack])
            targets_.append(targets[ix : ix + stack])
            targets2_.append(targets_2[ix : ix + stack])
        return images_, targets_, targets2_

    return images, targets


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
        label_to_count = Counter(dataset.labels)

        weights = [1.0 / label_to_count[ix] for ix in dataset.labels]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


def image_transforms(image_size=(84, 84), to_tensor=True):
    transforms_ = [
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.Grayscale(),
    ]

    if to_tensor:
        transforms_.append(transforms.ToTensor())
    return transforms.Compose(transforms_)


def transform_images(images, type="torch"):
    if type == "torch":

        transforms_ = image_transforms()
        images = torch.stack(
            [
                torch.stack([transforms_(image_).squeeze() for image_ in image_stack])
                for image_stack in images
            ]
        )

    elif type == "numpy":

        transforms_ = image_transforms(to_tensor=False)
        images = [
            np.stack([transforms_(image_) for image_ in image_stack])
            for image_stack in images
        ]

    return images


def reduce_gaze_stack(gaze_stack):
    gaze_pdfs = [gaze_pdf(gaze) for gaze in gaze_stack]
    pdf = np.sum(gaze_pdfs, axis=0)
    wpdf = pdf / np.sum(pdf)
    # print(torch.Tensor(wpdf).shape)
    # plt.imshow(wpdf)
    # plt.pause(12)
    # exit()

    return torch.Tensor(wpdf)


def gaze_pdf(gaze, gaze_count=1):
    pdfs_true = []
    gaze_range = [84, 84]  # w,h
    # gaze_range = [160.0, 210.0]  # w,h

    gaze_map = wpdf = np.zeros(gaze_range)

    gpts = np.multiply(gaze, gaze_range).astype(int)
    gpts = np.clip(gpts, 0, 83).astype(int)

    x, y = np.mgrid[0 : gaze_range[1] : 1, 0 : gaze_range[0] : 1]
    pos = np.dstack((x, y))
    if gaze_count != -1:
        gpts = gpts[-gaze_count:]

    for gpt in gpts:
        rv = multivariate_normal(
            mean=gpt[::-1], cov=[[2.85 * 2.85, 0], [0, 2.92 * 2.92]]
        )
        pdfs_true.append(rv.pdf(pos))
    pdf = np.sum(pdfs_true, axis=0)
    wpdf = pdf / np.sum(pdf)
    gaze_map = wpdf
    # assert abs(np.sum(wpdf) - 1) <= 1e-2, print(np.sum(wpdf))

    # for gpt in gpts:
    #     gaze_map[gpt[1], gpt[0]] = 1
    # gaze_map = gaze_map/np.sum(gaze_map)
    # draw_figs(wpdf, gazes=gaze_map)

    return gaze_map
