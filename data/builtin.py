"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Here we only register the few-shot datasets and complete COCO, PascalVOC and
LVIS have been handled by the builtin datasets in detectron2.
"""

import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.lvis import register_lvis_instances

from .builtin_meta import _get_builtin_metadata
from .register_o365 import register_all_o365
from .register_openimages import register_all_oi

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": (
        "coco/train2014",
        "coco/annotations/instances_train2014.json",
    ),
    "coco_2014_val": (
        "coco/val2014",
        "coco/annotations/instances_val2014.json",
    ),
    "coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/instances_minival2014.json",
    ),
    "coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/instances_minival2014_100.json",
    ),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": (
        "coco/train2017",
        "coco/annotations/instances_train2017.json",
    ),
    "coco_2017_val": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
    ),
    "coco_2017_test": (
        "coco/test2017",
        "coco/annotations/image_info_test2017.json",
    ),
    "coco_2017_test-dev": (
        "coco/test2017",
        "coco/annotations/image_info_test-dev2017.json",
    ),
    "coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/instances_val2017_100.json",
    ),
}

# === [export from FsDet] ===
import contextlib
import io
import os

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from .file_io import PathManager
from pycocotools.coco import COCO
def load_coco_json(json_file, image_root, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    is_shots = "shot" in dataset_name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "cocosplit")
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-2].split("shot")[0]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = dataset_name.split("_")[-1].split("shot")[0]
        for idx, cls in enumerate(metadata["thing_classes"]):
            json_file = os.path.join(
                split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls)
            )
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
    else:
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))
    id_map = metadata["thing_dataset_id_to_contiguous_id"]

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]

    if is_shots:
        for _, fileids_ in fileids.items(): # per category
            dicts = []
            for (img_dict, anno_dict_list) in fileids_: # 1 img, y annotations
                for anno in anno_dict_list: # per annotation
                    record = {}
                    record["file_name"] = os.path.join(
                        image_root, img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
    else:
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]] # filter out classes not in metadata here (80 -> 60)
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def register_meta_coco(name, metadata, imgdir, annofile):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(annofile, imgdir, metadata, name),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )
# ===

def register_all_coco(root="datasets"):
    # register meta datasets
    METASPLITS = [
        (
            "coco_trainval_all",
            "coco/trainval2014",
            "cocosplit/datasplit/trainvalno5k.json",
        ),
        (
            "coco_trainval_base",
            "coco/trainval2014",
            "cocosplit/datasplit/trainvalno5k.json",
        ),
        # ("coco_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        # ("coco_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        # ("coco_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_all", "coco/trainval2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_base", "coco/trainval2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_novel", "coco/trainval2014", "cocosplit/datasplit/5k.json"),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# ==== Predefined datasets and splits for LVIS ==========

# _PREDEFINED_SPLITS_LVIS = {
#     "lvis_v0.5": {
#         # "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
#         "lvis_v0.5_train_freq": (
#             "coco/train2017",
#             "lvis/lvis_v0.5_train_freq.json",
#         ),
#         "lvis_v0.5_train_common": (
#             "coco/train2017",
#             "lvis/lvis_v0.5_train_common.json",
#         ),
#         "lvis_v0.5_train_rare": (
#             "coco/train2017",
#             "lvis/lvis_v0.5_train_rare.json",
#         ),
#         # "lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
#         # "lvis_v0.5_val_rand_100": (
#         #     "coco/val2017",
#         #     "lvis/lvis_v0.5_val_rand_100.json",
#         # ),
#         # "lvis_v0.5_test": (
#         #     "coco/test2017",
#         #     "lvis/lvis_v0.5_image_info_test.json",
#         # ),
#     },
# }


# def register_all_lvis(root="datasets"):
#     for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
#         for key, (image_root, json_file) in splits_per_dataset.items():
#             # Assume pre-defined datasets live in `./datasets`.
#             register_lvis_instances(
#                 key,
#                 _get_builtin_metadata(dataset_name),
#                 os.path.join(root, json_file)
#                 if "://" not in json_file
#                 else json_file,
#                 os.path.join(root, image_root),
#             )

#     # register meta datasets
#     METASPLITS = [
#         (
#             "lvis_v0.5_train_shots",
#             "coco/train2017",
#             "lvissplit/lvis_shots.json",
#         ),
#         (
#             "lvis_v0.5_train_rare_novel",
#             "coco/train2017",
#             "lvis/lvis_v0.5_train_rare.json",
#         ),
#         ("lvis_v0.5_val_novel", "coco/val2017", "lvis/lvis_v0.5_val.json"),
#     ]

#     for name, image_root, json_file in METASPLITS:
#         dataset_name = "lvis_v0.5_fewshot" if "novel" in name else "lvis_v0.5"
#         register_meta_lvis(
#             name,
#             _get_builtin_metadata(dataset_name),
#             os.path.join(root, json_file)
#             if "://" not in json_file
#             else json_file,
#             os.path.join(root, image_root),
#         )


# # ==== Predefined splits for PASCAL VOC ===========
# def register_all_pascal_voc(root="datasets"):
#     # SPLITS = [
#     #     ("voc_2007_trainval", "VOC2007", "trainval"),
#     #     ("voc_2007_train", "VOC2007", "train"),
#     #     ("voc_2007_val", "VOC2007", "val"),
#     #     ("voc_2007_test", "VOC2007", "test"),
#     #     ("voc_2012_trainval", "VOC2012", "trainval"),
#     #     ("voc_2012_train", "VOC2012", "train"),
#     #     ("voc_2012_val", "VOC2012", "val"),
#     # ]
#     # for name, dirname, split in SPLITS:
#     #     year = 2007 if "2007" in name else 2012
#     #     register_pascal_voc(name, os.path.join(root, dirname), split, year)
#     #     MetadataCatalog.get(name).evaluator_type = "pascal_voc"

#     # register meta datasets
#     METASPLITS = [
#         ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
#         ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
#         ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
#         ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
#         ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
#         ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
#         ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
#         ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
#         ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
#         ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
#         ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
#         ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
#         ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
#         ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
#         ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
#         ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
#         ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
#         ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
#         ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
#         ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
#         ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
#     ]

#     # register small meta datasets for fine-tuning stage
#     for prefix in ["all", "novel"]:
#         for sid in range(1, 4):
#             for shot in [1, 2, 3, 5, 10]:
#                 for year in [2007, 2012]:
#                     for seed in range(100):
#                         seed = "" if seed == 0 else "_seed{}".format(seed)
#                         name = "voc_{}_trainval_{}{}_{}shot{}".format(
#                             year, prefix, sid, shot, seed
#                         )
#                         dirname = "VOC{}".format(year)
#                         img_file = "{}_{}shot_split_{}_trainval".format(
#                             prefix, shot, sid
#                         )
#                         keepclasses = (
#                             "base_novel_{}".format(sid)
#                             if prefix == "all"
#                             else "novel{}".format(sid)
#                         )
#                         METASPLITS.append(
#                             (name, dirname, img_file, keepclasses, sid)
#                         )

#     for name, dirname, split, keepclasses, sid in METASPLITS:
#         year = 2007 if "2007" in name else 2012
#         register_meta_pascal_voc(
#             name,
#             _get_builtin_metadata("pascal_voc_fewshot"),
#             os.path.join(root, dirname),
#             split,
#             year,
#             keepclasses,
#             sid,
#         )
#         MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
register_all_coco()
# register_all_lvis()
# register_all_pascal_voc()
register_all_o365()
register_all_oi()
