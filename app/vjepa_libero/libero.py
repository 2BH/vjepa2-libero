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
    world_size=1,
    camera_key="image",
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
):
    repo_id = "physical-intelligence/libero"

    delta_timestamps = {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        camera_key: [i*0.3 for i in range(16)],
        # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        "state": [i*0.3 for i in range(16)],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "actions": [i*0.3 for i in range(15)],
    }

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, episodes=[i for i in range(1200)])
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
