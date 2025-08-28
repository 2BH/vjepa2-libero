# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

_GLOBAL_SEED = 0
logger = getLogger()


def init_libero_data(
    batch_size,
    rank=0,
    input_length=16,
    train_episodes=1200,
    world_size=1,
    camera_key="image",
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
):
    repo_id = "physical-intelligence/libero"

    delta_timestamps = {
        camera_key: [i*0.3 for i in range(input_length)],
        "state": [i*0.3 for i in range(input_length)],
        "actions": [(i+1)*0.3 for i in range(input_length-1)],
    }

    if isinstance(train_episodes, int):
        train_episodes = [i for i in range(train_episodes)]
    elif isinstance(train_episodes, list):
        train_episodes = train_episodes
    else:
        raise ValueError(f"train_episodes must be an int or a list, got {type(train_episodes)}")

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, episodes=train_episodes)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler
