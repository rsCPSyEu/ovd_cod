# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import numpy as np
import math

import torch
import torch.utils.data as torchdata
from collections import defaultdict

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data import (
    load_proposals_into_dataset,
    print_instances_class_histogram,
)
from detectron2.data.build import (
    filter_images_with_only_crowd_annotations,
    filter_images_with_few_keypoints,
)

def repeat_factors_from_source_frequency(dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors based on dataset_source frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor:
                the i-th element is the repeat factor for the dataset image at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            # cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            cat_ids = {dataset_dict['source_idx'] for _ in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            # cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            cat_ids = {dataset_dict['source_idx'] for _ in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)


def get_detection_dataset_dicts_on_simple_concat(
    names,
    filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.
    Compared to get_detection_dataset_dicts function, this function supports multiple dataset sources.
    Speficially, it concatenates multiple datasets into one dataset.
    The labels spaces will be mapped into a single label space with a simple concatnations : 
    i.e., new label space = [c11, c12, c13, ..., c1i, c21, c22, c23, ..., c2j, ...], where i,j are the number of classes in dataset1, dataset2, ...

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        raise NotImplementedError("MultiDataset does not support torch Dataset!")
        # return torchdata.ConcatDataset(dataset_dicts)

    # assign source number to each data point
    for didx, data in enumerate(dataset_dicts):
        [d.update({'source_idx': didx}) for d in data]

        anns = [d['annotations'] for d in data]
        anns = list(itertools.chain.from_iterable(anns))
        used_cat_ids = np.unique([a['category_id'] for a in anns])
        print('For dataset {} | range of categody_id: [{}, {}] | len={}'.format(didx, min(used_cat_ids), max(used_cat_ids), len(used_cat_ids)))

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = all(["annotations" in d for d in dataset_dicts])
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if check_consistency and has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            # check_metadata_consistency("thing_classes", names) # no check for multi-dataset
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass
    
    # update assigned category_id
    label_space_per_dataset = [MetadataCatalog.get(n).thing_classes for n in names]
    num_labels = [0] + [len(l) for l in label_space_per_dataset]
    cum_num_labels = np.cumsum(num_labels)[:-1] # [0, num_cls_of_dataset1, num_cls_of_datase1+2, ...]
    for d in dataset_dicts:
        source_idx = d['source_idx']
        [a.update({
            'category_id': a['category_id']+cum_num_labels[source_idx], 'org_category_id': a['category_id']
        }) for a in d['annotations']]

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts