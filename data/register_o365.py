import os

from detectron2.data.datasets import register_coco_instances, load_coco_json
# from detectron2.data import MetadataCatalog, DatasetCatalog

data_root = 'datasets/o365'

def register_all_o365():
    # train
    register_coco_instances(
        'object365_06m_train',
        {}, 
        os.path.join(data_root, 'objects365_train_correctpath.json'), # annotation file # objects365_train_correctpath
        os.path.join(data_root), # image root
    )
    register_coco_instances(
        'object365_06m_val',
        {}, 
        os.path.join(data_root, 'objects365_val_correctpath.json'), # annotation file
        os.path.join(data_root), # image root
    )
    register_coco_instances(
        'object365_06m_val_debug',
        {}, 
        os.path.join(data_root, 'objects365_val_correctpath_subset4debug.json'), # annotation file
        os.path.join(data_root), # image root
    )
    
    # also register merged MSCOCO+Object365
    # This dataset contains X images from coco and O365 each. 
    # For category alignment, all categories in coco datasets except for 'sports ball' category are mapped into 365 categories in object365 dataset.
    # The annotations of 'sports ball' category in coco dataset are kept, wherese they are flagged as 'iscrowd' to be igrnored during training.
    for ratio in ['001', '010', '030', '050', '100']:   
        register_coco_instances(
            'mix_coco_o365_train_{}'.format(ratio),
            {}, 
            os.path.join('datasets', 'mix_coco_o365/coco_100k_o365_100k_{}.json'.format(ratio)), # annotation file # objects365_train_correctpath
            os.path.join('datasets'), # image root
        )