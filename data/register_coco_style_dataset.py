import os

from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog

from .custom_dataset import load_ft_json

def try_to_find(file, return_dir=False, search_path=['./datasets']):
    if not file:
        return file

    if file.startswith('catalog://'):
        return file

    DATASET_PATH = ['./']
    if 'DATASET' in os.environ:
        DATASET_PATH.append(os.environ['DATASET'])
    DATASET_PATH += search_path

    for path in DATASET_PATH:
        if os.path.exists(os.path.join(path, file)):
            if return_dir:
                return path
            else:
                return os.path.join(path, file)

    print('Cannot find {} in {}'.format(file, DATASET_PATH))
    exit(1)
    

def register_ft_instances(cfg, name, metadata, json_file, image_root, few_shot=-1, num_copy=-1, is_train=False):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        cfg (CfgNode): config instance
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ft_json(cfg, json_file, image_root, name, few_shot=few_shot, num_copy=num_copy, is_train=is_train))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )
    
def register_custom_data(cfg):
    # register custom dataset for finetuning
    for name, is_train in [(cfg.DATASETS.TRAIN, True), (cfg.DATASETS.TEST, False), (('test',), False)]: # forcebly add test dataset
        assert len(name) == 1, "Concat dataset is not implemented yet"
        name = name[0]
        data_info = cfg.DATASETS.REGISTER[name]
        img_dir = data_info['img_dir']
        ann_file = data_info['ann_file']
        
        ann_root = try_to_find(ann_file, return_dir=True)
        img_root = try_to_find(img_dir, return_dir=True)
        
        ann_file = os.path.join(ann_root, ann_file)
        img_dir = os.path.join(img_root, img_dir)
        
        if all([e is not None for e in cfg.FEWSHOT.SHOT_EPOCH_COPY]): # default: (None,None,None)
            custom_shot = int(cfg.FEWSHOT.SHOT_EPOCH_COPY[0])
            custom_epoch = int(cfg.FEWSHOT.SHOT_EPOCH_COPY[1])
            custom_copy = int(cfg.FEWSHOT.SHOT_EPOCH_COPY[2])
        else:
            # if is_train == False, dataset do not appply few-shot setting. i.e., load all images for evaluation.
            custom_shot = -1
            custom_copy = -1
        
        # if custom_shot <= 0, load all images (i.e., full-shot training)
        register_ft_instances(cfg, name, {}, ann_file, img_dir, few_shot=custom_shot, num_copy=custom_copy, is_train=is_train)
    return