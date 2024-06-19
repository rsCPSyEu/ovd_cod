# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import logging
import random
import os
import time
from copy import copy
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer

from detectron2.utils import comm
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
The code is based on detectron2/data/datasets/coco.py
"""

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')
# logger.setLevel(logging.INFO)

from logging import getLogger, StreamHandler
logger = getLogger(__name__)
# handler = StreamHandler()
# handler.setLevel(logging.DEBUG)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(handler)
# setup_logger()

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):    
    # filter out annotations which iscrowd=True
    anno = [a for a in anno if a.get("iscrowd", 0) == 0]
    
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # # for keypoint detection tasks, only consider valid images those
    # # containing at least min_keypoints_per_image
    # if _count_visible_keypoints(anno) >= cfg.DATALOADER.MIN_KPS_PER_IMS:
    #     return True
    return False


def get_fewshot_imgids(cfg, coco_api, all_img_ids, few_shot=-1):
    '''
    NOTE: This function is only for GLIP style's fewshot setting.
    Randomly select few_shot images from img_ids.
    
    coco_api: pycocotools object
    all_img_ids: list of img_ids
    few_shot: int, number of varioation of images to be selected        
    '''
    few_img_ids = []
    category_ids = [v['id'] for v in coco_api.cats.values()]
    category_id_range = range(min(category_ids), max(category_ids)+1)
    num_anns_each_cat = {cid: len(coco_api.getAnnIds(catIds=[cid])) for cid in category_id_range} # max number of annotations for each category
    # cats_freq = [few_shot]*max(list(coco_api.cats.keys()))
    cats_freq = [min(few_shot, num_anns_each_cat[cid]) for cid in category_id_range] # If the total number of annotations in the dataset is less than specified few_shot, set it as the number of maximum annotations.
    
    # apply rundom shuffle for self.ids
    if cfg.FEWSHOT.RUN_SEEDv1 >= 0:
        random.seed(cfg.FEWSHOT.RUN_SEEDv1)
        shuffle_ids = copy(all_img_ids)
        random.shuffle(shuffle_ids)
        all_img_ids = shuffle_ids
    
    for img_id in all_img_ids:
        if isinstance(img_id, str):
            ann_ids = coco_api.getAnnIds(imgIds=[img_id], iscrowd=None)
        else:
            ann_ids = coco_api.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco_api.loadAnns(ann_ids)
        cat = set([ann['category_id'] for ann in anno]) #set/tuple corresponde to instance/image level
        is_needed = sum([cats_freq[c-1]>0 for c in cat])
        
        if is_needed:
            few_img_ids.append(img_id)
            for c in cat:
                cats_freq[c-1] -= 1
    print('Loaded {} few-shot samples from {} images.'.format(len(few_img_ids), len(all_img_ids)))
    print('rank{}: Sampled few_img_ids={}(#{})'.format(comm.get_rank(), few_img_ids, len(few_img_ids)))
    with open(os.path.join(cfg.OUTPUT_DIR, 'fewshot_img_ids.txt.{}'.format(comm.get_rank())), 'w') as f:
        lines = ', '.join(map(str, few_img_ids))
        f.write('{}\n'.format(lines))
    comm.synchronize()
    time.sleep(3)
    return few_img_ids


def load_ft_json(cfg, json_file, image_root, dataset_name=None, extra_annotation_keys=None, few_shot=-1, num_copy=-1, is_train=False):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        cfg (CfgNode): config instance
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO
    
    # === Load few-shot sampled json file (FSOD style)
    loaded_new_fsod_json_file = False
    if few_shot>0 and is_train and cfg.FEWSHOT.RUN_SEEDv2 >= 0:
        # overwrite json_file
        # e.g., data_root = /path/to/dataset/ODinW/fewshot_annotation_v2/aquarium
        assert few_shot > 0, 'Please specify few_shot in config file'
        assert cfg.FEWSHOT.RUN_SEEDv1 < 0, 'Please do NOT turn on RUN_SEEDv1 in config file, when you use RUN_SEEDv2.'
        assert cfg.FEWSHOT.v2_DATA_ROOT != '', 'Please specify v2_DATA_ROOT in config file'
        new_json_file = os.path.join(cfg.FEWSHOT.v2_DATA_ROOT, 'train_{}_shot_seed_{}.json'.format(few_shot, cfg.FEWSHOT.RUN_SEEDv2))
        assert os.path.exists(new_json_file), 'File {} does not exist.'.format(new_json_file)
        logger.warning('Replace jsonfile {} with {}'.format(json_file, new_json_file))
        json_file = new_json_file
        loaded_new_fsod_json_file = True
    # === 
    
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    
    # Filter out invalid images, such as those without annotations (turn on only for train)
    if is_train:
        num_before = len(img_ids)
        new_ids = []
        for img_id in img_ids:
            if isinstance(img_id, str):
                ann_ids = coco_api.getAnnIds(imgIds=[img_id], iscrowd=None)
            else:
                ann_ids = coco_api.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = coco_api.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                new_ids.append(img_id)
        num_after = len(new_ids)
        img_ids = new_ids
        logger.info("Removed {} images with no usable annotations. {} images left.".format(num_before - num_after, num_after))
    
    # if few_shot > 0 and is_train and not in FSOD style:
    if few_shot > 0 and is_train and not loaded_new_fsod_json_file:
        # sample to get few-shot images (GLIP style)
        img_ids = get_fewshot_imgids(cfg, coco_api, img_ids, few_shot=few_shot)
    
    
    # For validation set: 
    # To make the training process faster, use only a part of the validation set.
    if not is_train and ('val' in dataset_name) and cfg.DATASETS.MAX_VAL_IMAGES > 0:
        if len(img_ids) > cfg.DATASETS.MAX_VAL_IMAGES:
            before_num = len(img_ids)
            img_ids = img_ids[:cfg.DATASETS.MAX_VAL_IMAGES]
            logger.info('Loaded {} validation images from {}.'.format(cfg.DATASETS.MAX_VAL_IMAGES, before_num))
        else:
            logger.info('Loaded {} validation images from {}.'.format(len(img_ids), len(img_ids)))
    
    
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    assert len(imgs) == len(anns), "len(imgs) != len(anns) in {}".format(json_file)
    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )        
    return dataset_dicts