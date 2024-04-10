import os
from yaml import safe_load
import numpy as np
from tqdm import tqdm
from subprocess import call
import h5py
import torch
import tarfile

from .data_loaders import load_images_action_gaze_data
from .data_utils import (
    get_game_entries_,
    process_gaze_data,
    transform_images,
    reduce_gaze_stack,
)


def create_interim_files(game, config):
    # getting variables from the config files
    raw_data_directory = config["raw_data_dir"]
    valid_actions = config["valid_actions"][game]
    interm_data_directory = config["interim_data_dir"]

    # Getting the game run names, game run directories, and corresponding gaze files
    game_runs, game_runs_dirs, game_runs_gazes = get_game_entries_(
        os.path.join(raw_data_directory, game)
    )

    # Make interm file directory if it does not exist
    interim_game_dir = os.path.join(interm_data_directory, game)
    if not os.path.exists(interim_game_dir):
        os.makedirs(interim_game_dir)

    # Loop through all game runs
    for game_run, game_run_dir, game_run_gaze in zip(
        game_runs, game_runs_dirs, game_runs_gazes
    ):

        # extract and save all .tar.bz2 (frame) files
        game_run_read_path = os.path.join(game_run_dir, game_run + ".tar.bz2")
        tar = tarfile.open(game_run_read_path, "r:bz2")
        tar.extractall(interim_game_dir)

        # extract and save gaze data
        gaze_read_file = os.path.join(game_run_dir, game_run_gaze)
        gaze_save_file = os.path.join(
            interim_game_dir, game_run, game_run + "_gaze_data.csv"
        )
        process_gaze_data(gaze_read_file, gaze_save_file, valid_actions)


def create_processed_data(
    config,
    stack=1,
    stack_type="",
    stacking_skip=1,
    from_ix=0,
    till_ix=-1,
    game="breakout",
    data_types=["frames", "actions", "gazes"],
):
    """Loads data from all the game runs in the src/data/interim  directory, and creates a hdf file in the src/data/processed directory.


    Args:
    ----
    data_types -- types of data to save, contains atleast on of the following
                ['frames', 'actions', 'gazes', 'fused_gazes',' gazes_fused_noop']

    stack -- number of frames in the stack

    stacking_skip -- Number of frames to skip while stacking

    from_ix --  starting index in the data, default is first, 0

    till_ix -- last index of the the data to be considered, default is last ,-1

    game : game to load the data from, directory of game runs

    Returns:
    ----
    None
    """
    # defines the interm and processed data paths
    game_interm_dir = os.path.join(config["interim_data_dir"], game)
    game_runs = os.listdir(game_interm_dir)
    game_h5_file = os.path.join(config["proccessed_data_dir"], game + ".hdf5")
    game_h5_file = h5py.File(game_h5_file, "w")

    for game_run in game_runs:
        # loads all images,actions and gazes
        images_, actions_, gazes_ = load_images_action_gaze_data(
            config, stack, stack_type, stacking_skip, from_ix, till_ix, game, game_run
        )
        # transforms images to gray scale 84 x 84 then a stack of torch tensors
        images_ = transform_images(images_, type="torch")

        group = game_h5_file.create_group(game_run)

        # potentially need to change this below
        gazes = torch.stack([reduce_gaze_stack(gaze_stack) for gaze_stack in gazes_])
        images_ = images_.numpy()
        gazes = gazes.numpy()

        group.create_dataset(
            "images",
            data=images_,
            compression=config["HDF_CMP_TYPE"],
            compression_opts=config["HDF_CMP_LEVEL"],
        )
        group.create_dataset(
            "actions",
            data=actions_,
            compression=config["HDF_CMP_TYPE"],
            compression_opts=config["HDF_CMP_LEVEL"],
        )
        group.create_dataset(
            "gazes",
            data=gazes,
            compression=config["HDF_CMP_TYPE"],
            compression_opts=config["HDF_CMP_LEVEL"],
        )

        del gazes, images_, actions_  # , gazes_fused_noop

    game_h5_file.close()


def combine_processed_data(game):
    """Reads the specified hdf5 file, and combines all the groups into a single combined group in the same file.


    Args:
    ----
    game -- name of the hdf5 file to combine, assumed to be in processed directory, without the extension

    Returns:
    ----
    None

    """

    gaze_out_h5_file = os.path.join(PROC_DATA_DIR, game + ".hdf5")
    gaze_h5_file = h5py.File(gaze_out_h5_file, "a")

    groups = list(gaze_h5_file.keys())
    if not "combined" in groups:
        all_group = gaze_h5_file.create_group("combined")
    all_group = gaze_h5_file["combined"]
    data = list(gaze_h5_file[groups[0]].keys())

    for datum in tqdm(data):
        max_shape_datum = (
            sum(
                [
                    gaze_h5_file[group][datum].shape[0]
                    for group in groups
                    if group != "combined"
                ]
            ),
            *gaze_h5_file[groups[0]][datum].shape[1:],
        )
        print(max_shape_datum, datum)
        all_group.create_dataset(
            datum,
            data=gaze_h5_file[groups[0]][datum][:],
            maxshape=max_shape_datum,
            compression=config["HDF_CMP_TYPE"],
        )

        for group in tqdm(groups[1:]):
            gaze_h5_file["combined"][datum].resize(
                gaze_h5_file["combined"][datum].shape[0]
                + gaze_h5_file[group][datum].shape[0],
                axis=0,
            )
            gaze_h5_file["combined"][datum][
                gaze_h5_file["combined"][datum].shape[0] :, :
            ] = gaze_h5_file[group][datum]

    gaze_h5_file.close()
