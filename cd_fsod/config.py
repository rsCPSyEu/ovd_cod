from detectron2.config import CfgNode as CN


def add_cdfsod_config(cfg):
    """
    Add config.
    """
    cfg.TEST.VAL_LOSS = True # 

    cfg.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0 # 
    cfg.MODEL.RPN.LOSS = "CrossEntropy" # 
    cfg.MODEL.ROI_HEADS.LOSS = "CrossEntropy" # 

    cfg.SOLVER.IMG_PER_BATCH_LABEL = 1
    cfg.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    cfg.SOLVER.FACTOR_LIST = (1,)

    cfg.TEST.EVALUATOR = "COCOeval"

    cfg.KD = CN()

    # Output dimension of the MLP projector after `res5` block
    cfg.KD.MLP_DIM = 128

    # Semi-supervised training
    cfg.KD.BBOX_THRESHOLD = 0.7
    cfg.KD.PSEUDO_BBOX_SAMPLE = "thresholding"
    cfg.KD.TEACHER_UPDATE_ITER = 1
    cfg.KD.BURN_UP_STEP = 12000
    cfg.KD.EMA_KEEP_RATE = 0.0
    cfg.KD.LOSS_WEIGHT = 4.0
    cfg.KD.LOSS_WEIGHT_TYPE = "standard"
    
    cfg.KD.DISABLE_TEACHER_EVAL = False

    cfg.EMAMODEL = CN()
    cfg.EMAMODEL.SUPcfgONSIST = True
    
