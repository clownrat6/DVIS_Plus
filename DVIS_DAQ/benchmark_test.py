# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
import copy
import itertools
import logging
import queue
import random
import threading
from collections.abc import MutableMapping, Sequence

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from train_net_video import Trainer, setup


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def main(args):
    cfg = setup(args)
    set_seed()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    model = model.eval()

    data_loader = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])

    for idx, inputs in enumerate(data_loader):
        with torch.amp.autocast("cuda"), torch.no_grad():
            # directly inference
            outputs = model(inputs)
        
        if idx == 5:
            break

    import time
    avg_time = []
    for idx, inputs in enumerate(data_loader):
        with torch.amp.autocast("cuda"), torch.no_grad():
            # online inference
            images = sum([x["image"] for x in inputs], [])
            images = images

            video_outputs = [None]
            for idx, image in enumerate(images):
                start = time.time()
                video_output = model.online_inference(idx, image[None])
                video_outputs.append(video_output)
                end = time.time()
                avg_time.append(end - start)
                if len(avg_time) > 5:
                    print("frame time:", sum(avg_time[5:]) / len(avg_time[5:]))
            video_outputs = video_outputs[1:]

        if idx == 10:
            break


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
