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
from torch.utils.data import Sampler
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
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

# Models
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from dvis_Plus import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    YTVISEvaluator,
    VPSEvaluator,
    VSSEvaluator,
    add_minvis_config,
    add_dvis_config,
    add_ctvis_config,
    get_detection_dataset_dicts,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
    UniYTVISEvaluator,
    SOTDatasetMapper,
)

from dvis_daq.config import add_daq_config

class TrainingSampler(Sampler):
    def __init__(self, size: int, shuffle: bool = True, seed: int = 42):
        self._size = size
        self._shuffle = shuffle

        self._seed = seed
        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self._world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), self._rank, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def to_cuda(packed_data):
    if isinstance(packed_data, bytes):
        return packed_data

    if isinstance(packed_data, torch.Tensor):
        packed_data = packed_data.to(device="cuda", non_blocking=True)
    elif isinstance(packed_data, (int, float, str, bool, complex)):
        packed_data = packed_data
    elif isinstance(packed_data, MutableMapping):
        for key, value in packed_data.items():
            packed_data[key] = to_cuda(value)
    elif isinstance(packed_data, Sequence):
        try:
            for i, value in enumerate(packed_data):
                packed_data[i] = to_cuda(value)
        except TypeError:
            pass
    return packed_data


class CUDADataLoader:

    def __init__(self, dataloader):
        self.dataloader = dataloader

        self.stream = torch.cuda.Stream() # create a new cuda stream in each process
        # setting a queue for storing prefetched data
        self.queue = queue.Queue(16)
        # 
        self.iter = dataloader.__iter__()
        # starting a new thread to prefetch data
        def data_to_cuda_then_queue():
            while True:
                try:
                    self.preload()
                except StopIteration:
                    break
            # NOTE: end flag for the queue
            self.queue.put(None)
        self.cuda_thread = threading.Thread(target=data_to_cuda_then_queue, args=())
        self.cuda_thread.daemon = True

        # NOTE: preload several batch of data
        (self.preload() for _ in range(8))
        self.cuda_thread.start()

    def preload(self):
        batch = next(self.iter)
        if batch is None:
            return None
        torch.cuda.current_stream().wait_stream(self.stream)  # wait tensor to put on GPU
        with torch.cuda.stream(self.stream):
            batch = to_cuda(batch)
            # batch = batch.to(device="cuda", non_blocking=True)
        self.queue.put(batch)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        # NOTE: __iter__ will be stopped when __next__ raises StopIteration 
        if next_item is None:
            raise StopIteration
        return next_item

    def __del__(self):
        # NOTE: clean up the thread
        try:
            self.cuda_thread.join(timeout=10)
        finally:
            if self.cuda_thread.is_alive():
                self.cuda_thread._stop()
        # NOTE: clean up the stream
        self.stream.synchronize()
        # NOTE: clean up the queue
        self.queue.queue.clear()


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name)
        os.makedirs(output_folder, exist_ok=True)

        evaluator_dict = {'vis': YTVISEvaluator, 'vss': VSSEvaluator, 'vps': VPSEvaluator}
        assert cfg.MODEL.MASK_FORMER.TEST.TASK in evaluator_dict.keys()
        return evaluator_dict[cfg.MODEL.MASK_FORMER.TEST.TASK](dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        assert len(cfg.DATASETS.DATASET_RATIO) == len(cfg.DATASETS.TRAIN) ==\
               len(cfg.DATASETS.DATASET_NEED_MAP) == len(cfg.DATASETS.DATASET_TYPE)
        mappers = []
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
            'image_instance': CocoClipDatasetMapper,
        }
        for d_i, (dataset_name, dataset_type, dataset_need_map) in \
                enumerate(zip(cfg.DATASETS.TRAIN, cfg.DATASETS.DATASET_TYPE, cfg.DATASETS.DATASET_NEED_MAP)):
            if dataset_type not in mapper_dict.keys():
                raise NotImplementedError
            _mapper = mapper_dict[dataset_type]
            mappers.append(
                _mapper(cfg, is_train=True, is_tgt=not dataset_need_map, src_dataset_name=dataset_name, )
            )
        assert len(mappers) > 0, "No dataset is chosen!"

        if len(mappers) == 1:
            mapper = mappers[0]
            train_loader = build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])
        else:
            loaders = [
                build_detection_train_loader(cfg, mapper=mapper, dataset_name=dataset_name)
                for mapper, dataset_name in zip(mappers, cfg.DATASETS.TRAIN)
            ]
            combined_data_loader = build_combined_loader(cfg, loaders, cfg.DATASETS.DATASET_RATIO)
            train_loader = combined_data_loader
        return CUDADataLoader(train_loader)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, dataset_type='video_instance'):
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
            'vos': SOTDatasetMapper,
        }
        if dataset_type not in mapper_dict.keys():
            raise NotImplementedError
        mapper = mapper_dict[dataset_type](cfg, is_train=False)
        test_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return CUDADataLoader(test_loader)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        skip_params = cfg.MODEL.VIDEO_HEAD.SKIP_PARAMS

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def test(self, cfg, model, evaluators=None):
        evaluators = [self.build_evaluator(cfg, dataset_name, output_folder=os.path.join(cfg.OUTPUT_DIR, f"model_{self.iter:07d}", dataset_name)) for dataset_name in cfg.DATASETS.TEST]
        return super().test(cfg, model, evaluators)

    @classmethod
    def eval(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))
        else:
            evaluators = [cls.build_evaluator(cfg, dataset_name, output_folder=os.path.join(cfg.OUTPUT_DIR, dataset_name)) for dataset_name in cfg.DATASETS.TEST]

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            evaluator = evaluators[idx]
            with torch.amp.autocast("cuda"):
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i,
                    dict), "Evaluator must return a dict on the main process. Got {} instead.".format(results_i)
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    add_dvis_config(cfg)
    add_ctvis_config(cfg)
    add_daq_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if 'OUTPUT_DIR' not in args.opts:
        work_dir_prefix = os.path.dirname(args.config_file).replace('configs/', '')
        work_dir_suffix = os.path.splitext(os.path.basename(args.config_file))[0]
        cfg.OUTPUT_DIR = f'work_dirs/{work_dir_prefix}/{work_dir_suffix}'
        if args.eval_only:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'eval')

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="minvis")
    return cfg


def main(args):
    set_seed(42)
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.eval(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
