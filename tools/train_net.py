"""
Dyhead Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
import itertools
import logging
import time
import numpy as np
# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.engine import SimpleTrainer, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, inference_on_dataset, verify_results, print_csv_format
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from dyhead import add_dyhead_config
from extra import add_extra_config
from defrcn import add_defrcn_config

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(Trainer, self).__init__(cfg)
        # logger = logging.getLogger("detectron2")
        # if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        #     setup_logger()
        # cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # model = self.build_model(cfg)
        # optimizer = self.build_optimizer(cfg, model)
        # data_loader = self.build_train_loader(cfg)
        # # Build DDP Model with find_unused_parameters to add flexibility.
        # if comm.get_world_size() > 1:
        #     model = DistributedDataParallel(
        #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
        #         find_unused_parameters=True
        #     )
        # self._trainer = SimpleTrainer(model, data_loader, optimizer)
        # self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # self.checkpointer = DetectionCheckpointer(
        #     # Assume you want to save checkpoints together with logs/statistics
        #     model,
        #     cfg.OUTPUT_DIR,
        #     optimizer=optimizer,
        #     scheduler=self.scheduler,
        # )
        # self.start_iter = 0
        # self.max_iter = cfg.SOLVER.MAX_ITER
        # self.cfg = cfg

        # self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, source_idx=None, is_fsdet=False):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if cfg.DATASETS.MULTI_SOURCE.ENABLE:
            from extra.evaluation.coco_evaluation import MultiSourceCOCOEvaluator
            num_labels = [0] + list(cfg.DATASETS.MULTI_SOURCE.NUM_CLASSES)
            cum_num_labels = np.cumsum(num_labels) # [0, num_cls_of_dataset1, num_cls_of_datase1+2, ...]
            return MultiSourceCOCOEvaluator(
                dataset_name, cfg, True, output_folder,
                source_idx=source_idx, index_diff=cum_num_labels[source_idx], valid_idx_range=cum_num_labels[source_idx:source_idx+2],
            )
        if is_fsdet:
            assert not cfg.DATASETS.MULTI_SOURCE.ENABLE, "Multi-Source is not supported in is_fsdet=True environment."
            from fsdet.evaluation.coco_evaluation import COCOEvaluator_FsDet
            return COCOEvaluator_FsDet(dataset_name, cfg, True, output_folder)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        logger = logging.getLogger(__name__)
        if cfg.DATASETS.MULTI_SOURCE.ENABLE:
            logger.info('Build train loader for Multi-Source : {}'.format(cfg.DATASETS.TRAIN))
            return cls.build_train_loader_for_multi_sources(cfg)
        
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
        return build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler)
    
    
    @classmethod
    def build_train_loader_for_multi_sources(cls, cfg):
        '''
        train loader with multi dataset sources 
        '''
        from data.build import get_detection_dataset_dicts_on_simple_concat, repeat_factors_from_source_frequency
        dataset = get_detection_dataset_dicts_on_simple_concat(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            check_consistency=False, # no check for multi soueces
        )
        
        mapper = None
        
        is_repeat_factor_sampler = (cfg.DATASETS.MULTI_SOURCE.RFS != '')
        if cfg.SEED!=-1:
            if is_repeat_factor_sampler:
                # [repeat factor sampling according to dataset source]
                if cfg.DATASETS.MULTI_SOURCE.RFS == 'source':
                    num_sources = cfg.DATASETS.TRAIN.__len__()
                    repeat_factors = repeat_factors_from_source_frequency(dataset, repeat_thresh=1/num_sources)
                # [repeat factor sampling according to categories]
                elif cfg.DATASETS.MULTI_SOURCE.RFS == 'category':
                    num_categories = cfg.MODEL.ATSS.NUM_CLASSES if cfg.MODEL.META_ARCHITECTURE=="ATSS" else cfg.MODEL.ROI_HEADS.NUM_CLASSES
                    repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(dataset, repeat_thresh=1/num_categories)
                else:
                    raise NotImplementedError                
                sampler = RepeatFactorTrainingSampler(repeat_factors, seed=cfg.SEED)
            else:
                sampler = TrainingSampler(len(dataset), seed=cfg.SEED)
        else:
            sampler = None
        return build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler)
    

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
                    evaluator = cls.build_evaluator(cfg, dataset_name, source_idx=idx, is_fsdet=cfg.DATASETS.IS_FSDET)
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
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dyhead_config(cfg)
    add_extra_config(cfg)
    add_defrcn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):    
    cfg = setup(args)

    # register more datasets
    import data.builtin as _UNUSED # inside this line, register o365 and openimages

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    # [debug]
    if dist.is_initialized():
        print('================================================')
        print('rank{}: nccl_socket_ifname: {}'.format(dist.get_rank(), os.environ.get("NCCL_SOCKET_IFNAME")))
        print('rank{}: NCCL_P2P_LEVEL: {}'.format(dist.get_rank(), os.environ.get("NCCL_P2P_LEVEL")))
        print('================================================')
    
    # # [debug for model size check] 
    # def show_model_params(model):
    #     from pprint import pprint
    #     out_file = 'model_params.log'
    #     param_size = 0
    #     with open(out_file, 'w') as f:
    #         for module_name, module in model.named_modules():
    #             total_params = sum(p.numel() for p in module.parameters(recurse=False))
    #             if total_params > 0:
    #                 print(f"{module_name}: {total_params} [{total_params/(1000**2)} MB] parameters", file=f)
    #             param_size += total_params
    #         print('Total model size: {:.3f} [{} MB]'.format(param_size, param_size/(1000**2)), file=f)
    #     return
    # show_model_params(trainer.model)
    
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
    # [default]
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=int(os.environ['AZ_BATCHAI_TASK_INDEX']) \
    #                     if 'AZ_BATCHAI_TASK_INDEX' in os.environ else args.machine_rank,
    #     dist_url="tcp://"+os.environ['AZ_BATCH_MASTER_NODE'] \
    #                     if 'AZ_BATCH_MASTER_NODE' in os.environ else args.dist_url,
    #     args=(args,),
    # )