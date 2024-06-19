from detectron2.config import CfgNode as CN


def add_extra_config(cfg):

    # extra configs for swin transformer
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False
    cfg.MODEL.SWINT.VERSION = 1
    cfg.MODEL.SWINT.OUT_NORM = True
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # extra configs for atss
    cfg.MODEL.ATSS = CN()
    cfg.MODEL.ATSS.TOPK = 9
    cfg.MODEL.ATSS.NUM_CLASSES = 80
    cfg.MODEL.ATSS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.ATSS.NUM_CONVS = 4
    cfg.MODEL.ATSS.CHANNELS = 256
    cfg.MODEL.ATSS.USE_GN = True

    cfg.MODEL.ATSS.IOU_THRESHOLDS = [0.4, 0.5]
    cfg.MODEL.ATSS.IOU_LABELS = [0, -1, 1]
    cfg.MODEL.ATSS.PRIOR_PROB = 0.01
    cfg.MODEL.ATSS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
    cfg.MODEL.ATSS.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.ATSS.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

    cfg.MODEL.ATSS.INFERENCE_TH = 0.05
    cfg.MODEL.ATSS.PRE_NMS_TOP_N = 1000
    cfg.MODEL.ATSS.NMS_TH = 0.6

    cfg.MODEL.ATSS.USE_COSSIM = False # If the model uses cosine similarity output for classification, set this to True.
    cfg.MODEL.ROI_HEADS.COSINE_SCALE = 20.0 # following FSOD's setting / if -1, learnable scale factor is used.

    cfg.SOLVER.OPTIMIZER = 'SGD'
    cfg.SOLVER.AUTO_TERMINATE_PATIENCE = -1
    
    # add register to define train/test dataset on yaml file directly
    cfg.DATASETS.REGISTER = CN(new_allowed=True)
    cfg.SOLVER.MAX_EPOCH = -1
    cfg.SOLVER.EPOCH_ITER = 'epoch' # training based on epoch or iteration
    
    cfg.TEST.EVAL_EPOCH = -1

    # fewshot finetune setting
    cfg.FEWSHOT = CN()
    cfg.FEWSHOT.BASE_CONFIG = '' # specify yaml file from GLIP repositry, if any
    cfg.FEWSHOT.SHOT_EPOCH_COPY = (None,None,None) # (epoch, copy, copy_epoch)
    cfg.FEWSHOT.RANDOM_SAMPLE = False # 
    cfg.FEWSHOT.FREEZE_METHOD = ''
    cfg.FEWSHOT.RUN_SEEDv1 = -1 # [For GLIP style's fewshot] if >= 0, shuffle original image_ids in data.custom_dataset.get_fewshot_imgids to add randomness.
    cfg.FEWSHOT.RUN_SEEDv2 = -1 # [For FSOD style's fewshot] if >= 0, use different fewshot samples. 
    cfg.FEWSHOT.v2_DATA_ROOT = ''

    # optimizer for finetuning
    cfg.SOLVER.BACKBONE_BODY_LR_FACTOR = 1.0
    cfg.SOLVER.WEIGHT_DECAY_NORM_FACTOR = 1.0
    cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.0
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.SOLVER.MODEL_EMA = 0.0
    
    # lr scheduler for finetuning
    cfg.SOLVER.MULTI_MAX_EPOCH = ()  # set different max epoch for different stage
    cfg.SOLVER.USE_AUTOSTEP = False
    cfg.SOLVER.STEP_PATIENCE = 5
    cfg.SOLVER.USE_COSINE = False
    cfg.SOLVER.MIN_LR = 0.000001
    
    # multi source setting
    cfg.DATASETS.MULTI_SOURCE = CN()
    cfg.DATASETS.MULTI_SOURCE.ENABLE = False # if true, use multi source dataset such as [coco, o365, openimages]
    cfg.DATASETS.MULTI_SOURCE.NUM_CLASSES = ()
    cfg.DATASETS.MULTI_SOURCE.RFS = '' # repeat factor sampling : choices = ['category', 'source']
    cfg.DATASETS.IS_FSDET = False # if true, use coco-split (base:60, novel=20) as datasets
    
    cfg.DATASETS.MAX_VAL_IMAGES = -1
        
    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.NAME = None
    cfg.WANDB.PROJECT = None
    cfg.WANDB.WANDB_RESUME_RUN_NAME = None