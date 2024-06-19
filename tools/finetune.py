"""
Dyhead Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
import itertools
import logging
import time
import json
# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import yaml 
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results

from dyhead import add_dyhead_config
from extra import add_extra_config
from defrcn import add_defrcn_config
from fsce import add_fsce_config
from cd_fsod import add_cdfsod_config

keep_testing = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*__floordiv__ is deprecated.*") # to prevent warnings

def adjust_num_classes_cfg(cfg, target_num_classes):
    for node_name in cfg:
        child = cfg[node_name]
        if child.__class__ != CfgNode:
            continue
        if hasattr(child, 'NUM_CLASSES'):
            child['NUM_CLASSES'] = target_num_classes
        child = adjust_num_classes_cfg(child, target_num_classes)
    return cfg


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dyhead_config(cfg)
    add_extra_config(cfg)
    add_defrcn_config(cfg)
    add_fsce_config(cfg)
    add_cdfsod_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
        
    is_fsdet = cfg.DATASETS.IS_FSDET
    
    # if some base config for few-shot setting is specified, extract some arguments 
    if not is_fsdet:
        assert len(cfg.FEWSHOT.BASE_CONFIG) > 0, 'cfg.FEWSHOT.BASE_CONFIG must be specified'
        cfg_fs = get_cfg()
        add_extra_config(cfg_fs)
        # cfg.merge_from_file(cfg.FEWSHOT.BASE_CONFIG)
        loaded_cfg = cfg_fs.load_yaml_with_base(cfg.FEWSHOT.BASE_CONFIG, allow_unsafe=True)
        cfg_fs = type(cfg_fs)(loaded_cfg)
        
        # what we need are DATASETS and NUM_CLASSES info.
        cfg.DATASETS.REGISTER = cfg_fs.DATASETS.REGISTER
        cfg.DATASETS.TRAIN = (cfg_fs.DATASETS.TRAIN.split('"')[1],) # expects 'train' 
        cfg.DATASETS.TEST = (cfg_fs.DATASETS.TEST.split('"')[1],) # expects 'val' or 'minival'
        
        target_num_classes = cfg_fs.MODEL.ATSS.NUM_CLASSES - 1 # NUM_CLASSES of GLIP's cfg (cfg_fs) contains background class, so we need to set it as -1.
        cfg = adjust_num_classes_cfg(cfg, target_num_classes)        
    
        cfg.freeze()
        default_setup(cfg, args)
        
        from data.register_coco_style_dataset import register_custom_data
        register_custom_data(cfg)
    else:
        cfg.freeze()
        default_setup(cfg, args)
    return cfg


def check_frozen(model):
    if not comm.is_main_process():
        return
    updates = []
    frozen = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            updates.append(name)
        else:
            frozen.append(name)
    print('======')
    print("[Updated parameters]: \n{}".format('\n'.join(updates)))
    print('======')
    print("[Frozen parameters]: \n{}".format('\n'.join(frozen)))
    return


def prepare_fsce_module(trainer, arch):
    if arch == 'GeneralizedRCNN':
        for n,p in trainer.model.named_parameters():
            if 'backbone.bottom_up' in n:
                p.requires_grad = False
    elif arch == 'ATSS':
        for n,p in trainer.model.named_parameters():
            if 'backbone.backbone.bottom_up' in n:
                p.requires_grad = False
    else:
        raise NotImplementedError('Unknown architecture: {}'.format(arch))
              

def freeze_module(trainer, keys, operation='freeze'):
    '''
    A Function to freeze the parameters of a module.
    If the named paremeter ends with at least one of the key names which are specified by "key", then it will be frozen.

    Args:
        trainer (detectron2.engine.trainer.Trainer): trainer object
        key (list(str)): list of module names which are frozen during training
        operation (str): 'freeze' or 'unfreeze', which are the operation applied to any modules that are matched with any keys
    '''        
    assert operation in ['freeze', 'unfreeze'], 'operation must be either "freeze" or "unfreeze"'
    operation = False if operation == 'freeze' else True
    assert type(keys) == list, 'key must be a list of str'
    for n,m in trainer.model.named_modules():
        if any([n.endswith(k) for k in keys]):
            # matched: do operation
            for p in m.parameters():
                p.requires_grad = operation
        else:
            # not matched: do opposite operation
            for p in m.parameters():
                p.requires_grad = (not operation)


def main(args):
    cfg = setup(args)

    if cfg.FEWSHOT.FREEZE_METHOD == 'distill_cdfsod':
        # initialize distill_cdfsod
        from cd_fsod.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
        from cd_fsod.modeling.proposal_generator.rpn import PseudoLabRPN
        from cd_fsod.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
        from cd_fsod.engine.trainer import KdTrainer as Trainer # replace ft_trainer.Trainer with cd_fsod.engine.trainer.KdTrainer
    else:
        from ft_trainer import Trainer
        
    import data.builtin as _UNUSED
    is_fsdet = cfg.DATASETS.IS_FSDET # if True, use cfg.DATASETS.TRAIN and TEST as they are, otherwise, modify in the following part.
    
    if args.eval_only:
        if not is_fsdet:
            cfg.defrost()
            cfg.DATASETS.TEST = ("test", ) # from ('val' or 'minival') to ('test',)
            cfg.freeze()
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, "inference"), exist_ok=True)
            with open(os.path.join(cfg.OUTPUT_DIR, "inference", "res_final.json"), "w") as fp:
                json.dump(res, fp)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    
    # [freeze weight accroding to arguments]
    # if len(cfg.FEWSHOT.FREEZE_METHOD) > 0:
    if cfg.FEWSHOT.FREEZE_METHOD == 'full_ft':
        # all parametes are trainable
        # cfg.MODEL.BACKBONE.FREEZE_AT = -1
        assert all([m.requires_grad for n,m in trainer.model.named_parameters()])
        pass
    elif cfg.FEWSHOT.FREEZE_METHOD == 'tfa': # Two-stage Finetune Approach (https://arxiv.org/abs/2003.06957)
        # freeze all parameters except for the last fc classifier and regressor
        if cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN':
            freeze_module(trainer, ['box_predictor.cls_score', 'box_predictor.bbox_pred'], operation='unfreeze')
        elif cfg.MODEL.META_ARCHITECTURE == 'ATSS':
            freeze_module(trainer, ['head.cls_logits', 'head.bbox_pred', 'head.centerness'], operation='unfreeze')
        else:
            raise NotImplementedError('Unknown meta architecture: {}'.format(cfg.MODEL.META_ARCHITECTURE))
    elif cfg.FEWSHOT.FREEZE_METHOD == 'tfa_rpn': # Two-stage Finetune Approach (https://arxiv.org/abs/2003.06957), but RPN is also trainable
        # freeze all parameters except for the last fc classifier and regressor
        if cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN':
            freeze_module(trainer, 
                          ['rpn_head.objectness_logits', 'rpn_head.anchor_deltas', 'box_predictor.cls_score', 'box_predictor.bbox_pred'], 
                          operation='unfreeze')
        else:
            raise NotImplementedError('Unknown meta architecture: {}'.format(cfg.MODEL.META_ARCHITECTURE))
    elif cfg.FEWSHOT.FREEZE_METHOD == 'freeze_ie': # Simply unfreeze Backbone Network
        assert all([m.requires_grad for n,m in trainer.model.named_parameters()])
        if cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN':
            prepare_fsce_module(trainer, cfg.MODEL.META_ARCHITECTURE) # repurpose the case of FSCE
        elif cfg.MODEL.META_ARCHITECTURE == 'ATSS':
            prepare_fsce_module(trainer, cfg.MODEL.META_ARCHITECTURE) # repurpose the case of FSCE
        else:
            raise NotImplementedError('Unknown meta architecture: {}'.format(cfg.MODEL.META_ARCHITECTURE))
    elif cfg.FEWSHOT.FREEZE_METHOD == 'fsce': # Contrastive Approach (https://arxiv.org/pdf/2103.05950)
        # freeze only backbone, and make FPN, RPN, and RoI trainable
        assert all([m.requires_grad for n,m in trainer.model.named_parameters()])
        if cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN':
            prepare_fsce_module(trainer, cfg.MODEL.META_ARCHITECTURE)
        # elif cfg.MODEL.META_ARCHITECTURE == 'ATSS':
        #     freeze_module(trainer, ['head.cls_logits', 'head.bbox_pred', 'head.centerness'], operation='unfreeze')
        else:
            raise NotImplementedError('Unknown meta architecture: {}'.format(cfg.MODEL.META_ARCHITECTURE))
    elif cfg.FEWSHOT.FREEZE_METHOD == 'defrcn': 
        # RPN is trainable, but the cls/reg head is frozen
        # All freeze/unfrezee operations are done in config
        # assert (not cfg.MODEL.RPN.FREEZE) and (cfg.MODEL.ROI_HEADS.FREEZE_FEAT) and (cfg.MODEL.ROI_HEADS.CLS_DROPOUT)
        pass
    elif cfg.FEWSHOT.FREEZE_METHOD == 'distill_cdfsod':
        pass
    else:
        raise NotImplementedError('Unknown freeze method: {}'.format(cfg.FEWSHOT.FREEZE_METHOD))

    # [check frozen parameters]
    check_frozen(trainer.model)    
    
    lines = []
    for n,m in trainer.model.named_parameters():
        lines.append('{} | {} | {}'.format(n, m.shape, m.requires_grad))
        
    _last_eval_results = trainer.train()
    
    # [skip the last evaluation on test set]
    if not keep_testing:
        return _last_eval_results
    # [Fsdet will use the setting where train=60 coco and test=20 coco]
    if not is_fsdet:
        cfg.defrost()
        cfg.DATASETS.TEST = ("test", ) # from ('val' or 'minival') to ('test',)
        cfg.freeze()
    # [reload model_best.pth]
    best_weight_path = os.path.join(cfg.OUTPUT_DIR, 'model_best.pth')
    if os.path.isfile(best_weight_path):
        DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'), resume=args.resume)
    else:
        print(' === === model_best.pth does not exist. Testing on the final model. === === ')
     
    res = Trainer.test(cfg, trainer.model)
    if comm.is_main_process():
        verify_results(cfg, res)
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, "inference"), exist_ok=True)
        with open(os.path.join(cfg.OUTPUT_DIR, "inference", "res_final.json"), "w") as fp:
            json.dump(res, fp)
             
    return res
    
    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=int(os.environ['AZ_BATCHAI_TASK_INDEX']) \
                        if 'AZ_BATCHAI_TASK_INDEX' in os.environ else args.machine_rank,
        dist_url="tcp://"+os.environ['AZ_BATCH_MASTER_NODE'] \
                        if 'AZ_BATCH_MASTER_NODE' in os.environ else args.dist_url,
        args=(args,),
    )