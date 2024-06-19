"""
Dyhead Training Script for Finetuning process.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
import itertools
import logging
import time
import math
# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from typing import Any, Dict, List, Set

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.utils.data as torchdata
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import build_detection_test_loader
from detectron2.data.samplers import TrainingSampler
from detectron2.data.build import get_detection_dataset_dicts, _train_loader_from_config
from detectron2.engine import SimpleTrainer, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, inference_on_dataset, verify_results, print_csv_format
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage
from detectron2.config import configurable
from detectron2.data.common import DatasetFromList, MapDataset

from hooks import AutoTerminateHook, AutoStepHook
from ft.solver import make_optimizer, make_lr_scheduler

from data.duplicate_dataset import DupDatasetFromList

@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader( # based on detectron2.data.build.build_detection_train_loader
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    num_duplication=-1, # number of duplication of datasets
):
    # from list to Dataset format
    if isinstance(dataset, list):
        if num_duplication > -1:
            args = {'lst': dataset, 'copy': False}
            dataset = DupDatasetFromList(fewshot_copy=num_duplication, **args)
        else:
            dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"

    from detectron2.data import build_batch_data_loader
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.logger = logger
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        
        data_loader = self.build_train_loader(cfg)
        # Build DDP Model with find_unused_parameters to add flexibility.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
                find_unused_parameters=True
            )
        
        # setup some options : e.g., eval period 
        cfg = self.setup_eval_period(cfg, data_loader)
        
        # [default implementation from detectron2]    
        # optimizer = self.build_optimizer(cfg, model)
        # self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # [or, custom implmenentation from GLIP]
        assert cfg.SOLVER.MAX_ITER == self.max_iter
        optimizer = make_optimizer(cfg, model) 
        self.scheduler = make_lr_scheduler(cfg, optimizer)
        
        self._trainer = SimpleTrainer(model, data_loader, optimizer)        
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.cfg = cfg


        # [for distllation training]
        if cfg.FEWSHOT.FREEZE_METHOD == 'distill_cdfsod':
            # create an teacher model
            model_teacher = self.build_model(self.cfg)
            self.model_teacher = model_teacher
            
            if not os.path.exists(cfg.OUTPUT_DIR.replace('student','teacher')):
                os.mkdir(cfg.OUTPUT_DIR.replace('student','teacher'))
                
            self.t_checkpointer = DetectionCheckpointer(
                model_teacher,
                cfg.OUTPUT_DIR.replace('student','teacher'),
                optimizer=optimizer,
                scheduler=self.scheduler,
            )
        
        # [add AutoTerminateHook here]
        val_metric = 'bbox_student/AP' if cfg.FEWSHOT.FREEZE_METHOD=='distill_cdfsod' else 'bbox/AP'
        hooks = self.build_hooks()
        
        if cfg.SOLVER.USE_AUTOSTEP:
            autostep_hook = AutoStepHook(self.scheduler, eval_period=cfg.TEST.EVAL_PERIOD, val_metric=val_metric)
            hooks.append(autostep_hook)
        
        if cfg.SOLVER.AUTO_TERMINATE_PATIENCE > 0:
            # if comm.is_main_process(): # add only main process, same as PeriodicCheckpointer
            auto_terminate_patience = cfg.SOLVER.AUTO_TERMINATE_PATIENCE
            auto_terminate_hook = AutoTerminateHook(
                auto_terminate_patience,
                cfg.TEST.EVAL_PERIOD, 
                self.checkpointer,
                val_metric=val_metric,
                mode='max',
                file_prefix='model_best',
                output_dir=cfg.OUTPUT_DIR,
            )                
            hooks.append(auto_terminate_hook)
            self.use_auto_terminate = True # activate to check auto termination
            # device = torch.device('cuda:{}'.format(comm.get_rank()))
            self.is_terminate = torch.Tensor([False]).to(torch.device('cuda')) # when switched into True, the training will be terminated
        else:
            self.use_auto_terminate = False
            self.is_terminate = torch.Tensor([False]).to(torch.device('cuda'))
        
        self.register_hooks(hooks)


    def setup_eval_period(self, cfg, data_loader):
        cfg.defrost()
        
        if all([e is not None for e in cfg.FEWSHOT.SHOT_EPOCH_COPY]):
            num_shot, max_epochs, num_copy, = cfg.FEWSHOT.SHOT_EPOCH_COPY
        else:
            max_epochs = cfg.SOLVER.MAX_EPOCH
            num_copy = 1
        
        assert hasattr(cfg.SOLVER, 'MAX_EPOCH'), 'Please define SOLVER.MAX_EPOCH in config file'
        # define max_iteration according to max_epoch
        self.max_epoch = max_epochs
        batch_size = cfg.SOLVER.IMS_PER_BATCH # sum of batch size for all GPUs : e.g., 8b * 4gpus = 32
        dataset_len  = len(data_loader.dataset.dataset.dataset._dataset)
        iter_per_epoch = dataset_len // batch_size
        self.iter_per_epoch = iter_per_epoch
        
        if cfg.SOLVER.EPOCH_ITER == 'epoch':
            # epoch base training
            max_iter = iter_per_epoch * max_epochs
            # max_iter = min(iter_per_epoch * max_epochs, iter_per_epoch) # if iter_per_epoch * max_epochs is too large, set an upper bound as 1 epoch
            # if max_iter > cfg.SOLVER.MAX_ITER:
            #     self.logger.info('*** max_iter is larger than cfg.SOLVER.MAX_ITER, so set max_iter as cfg.SOLVER.MAX_ITER.')
            #     max_iter = min(max_iter, cfg.SOLVER.MAX_ITER)
            self.max_iter = max_iter
            cfg.SOLVER.MAX_ITER = max_iter
            # set cfg.TEST.EVAL_PERIOD to make checkpointer (to save model) correctly
            cfg.TEST.EVAL_PERIOD = iter_per_epoch * cfg.TEST.EVAL_EPOCH # eval per epoch
            self.logger.info('Batch size = {}'.format(batch_size))
            self.logger.info('Num of images for epoch = {} data points * {} copies = {}'.format(dataset_len/num_copy, num_copy, dataset_len))
            self.logger.info("Iter per epoch = {} images // {} batch_size = {} iters".format(dataset_len, batch_size, iter_per_epoch))
            self.logger.info("Max iteration = {} iters * {} epochs = {} iters".format(iter_per_epoch, max_epochs, max_iter))
            self.logger.info("Eval Period = {} iter * {} epochs = {} iters".format(iter_per_epoch, cfg.TEST.EVAL_EPOCH, cfg.TEST.EVAL_PERIOD))
            self.logger.info("Checkpoint Period = {} iters".format(cfg.SOLVER.CHECKPOINT_PERIOD))
            
        elif cfg.SOLVER.EPOCH_ITER == 'iter':
            # iter base training
            max_iter = cfg.SOLVER.MAX_ITER
            self.max_iter = max_iter
            self.logger.info('Batch size = {}'.format(batch_size))
            self.logger.info('Num of images for epoch = {} data points * {} copies = {}'.format(dataset_len/num_copy, num_copy, dataset_len))
            self.logger.info("Iter per epoch = {} images // {} batch_size = {} iters".format(dataset_len, batch_size, iter_per_epoch))
            self.logger.info("Max iteration = {} iters".format(max_iter))
            self.logger.info("Eval Period = {} iters".format(cfg.TEST.EVAL_PERIOD))
            self.logger.info("Checkpoint Period = {} iters".format(cfg.SOLVER.CHECKPOINT_PERIOD))
        cfg.freeze()
        return cfg
           
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, is_fsdet=False):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if is_fsdet:
            from fsdet.evaluation.coco_evaluation import COCOEvaluator_FsDet
            return COCOEvaluator_FsDet(dataset_name, cfg, True, output_folder)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        mapper = None
        if cfg.SEED!=-1:
            sampler = TrainingSampler(len(dataset), seed=cfg.SEED)
        else:
            sampler = None
        return build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler, num_duplication=cfg.FEWSHOT.SHOT_EPOCH_COPY[2])

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = None
        return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                if weight_decay is None:
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY # None means following WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
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

    @classmethod
    def make_optimizer(self, cfg, model):
        '''
        Optimizer for finetuning, derived from GLIP implementation. (maskrcnn_benchmark/solver/build.py in GLIP)
        '''    
        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
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

        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            # different lr schedule
            if "language_backbone" in key:
                lr = cfg.SOLVER.LANG_LR

            if "backbone.body" in key and "language_backbone.body" not in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_BODY_LR_FACTOR

            if "bias" in key:
                lr *= cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

            if 'norm' in key or 'Norm' in key:
                weight_decay *= cfg.SOLVER.WEIGHT_DECAY_NORM_FACTOR
                print("Setting weight decay of {} to {}".format(key, weight_decay))

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if cfg.SOLVER.OPTIMIZER == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(params, lr, momentum=cfg.SOLVER.MOMENTUM)
        elif cfg.SOLVER.OPTIMIZER == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, lr)

        return optimizer


    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
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
        logger.setLevel(logging.INFO)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    # evaluator = cls.build_evaluator(cfg, dataset_name)
                    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                    # turn on is_fsdet if base_config is empty
                    evaluator = cls.build_evaluator(cfg, dataset_name, output_folder=output_folder, 
                                                    is_fsdet=cfg.DATASETS.IS_FSDET)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            # [defrcn]
            if cfg.TEST.PCB_ENABLE:
                from defrcn.evaluation.evaluator import inference_on_dataset as inference_on_dataset_defrcn
                results_i = inference_on_dataset_defrcn(model, data_loader, evaluator, cfg)
            else:
                results_i = inference_on_dataset(model, data_loader, evaluator)
                # with EventStorage() as strage:
                #     results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(results_i)
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
            
        if results.get('bbox', None) is not None:
            # convert nan into 0. score
            for k in results['bbox'].keys():
                if math.isnan(results['bbox'][k]):
                    results['bbox'][k] = 0. 
            print('results: {}'.format(results))
            
        return results
    
    
    def train_loop(self, start_iter, max_iter):       
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter        
        
        self.logger.info("Starting training from iteration {}".format(self.start_iter))
        self.logger.info("Max iteration = {}".format(self.max_iter))
        
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    # auto terminate if the hookd has 'is_terminate==True'
                    if self.use_auto_terminate and self.is_terminate:
                        break
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                self.logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
    
    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results