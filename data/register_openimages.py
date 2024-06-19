import os

from detectron2.data.datasets import register_coco_instances, load_coco_json
# from detectron2.data import MetadataCatalog, DatasetCatalog

data_root='datasets/openimages'

def register_all_oi():
    # train
    register_coco_instances(
        'openimages_train',
        {}, 
        os.path.join(data_root, 'annotations', 'openimages_v6_train_bbox.json'), # annotation file # objects365_train_correctpath
        os.path.join(data_root, 'train'), # image root
    )
    register_coco_instances(
        'openimages_val',
        {}, 
        os.path.join(data_root, 'annotations', 'openimages_v6_val_bbox.json'), # annotation file
        os.path.join(data_root, 'validation'), # image root
    )
    register_coco_instances(
        'openimages_test',
        {}, 
        os.path.join(data_root, 'annotations', 'openimages_v6_val_bbox.json'), # annotation file
        os.path.join(data_root, 'test'), # image root
    )