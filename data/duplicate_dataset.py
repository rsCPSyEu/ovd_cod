import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import random
import numpy as np

from detectron2.data.common import DatasetFromList

class DupDatasetFromList(DatasetFromList):
    '''
    Expected to overwrite on detectron2.data.common.DatasetFromList
    By specifying fewshot_copy, the dataset will be duplicated with 'fewshot_copy' times.
    '''
    def __init__(self, fewshot_copy, **kwargs):
        super(DupDatasetFromList, self).__init__(**kwargs)
        self.fewshot_copy = fewshot_copy
        self.length = super(DupDatasetFromList, self).__len__() # num data points before duplication

    def __len__(self):
        return self.fewshot_copy * self.length

    def __getitem__(self, index):
        true_index = index % self.length
        return super(DupDatasetFromList, self).__getitem__(true_index)

    # def get_img_info(self, index):
    #     true_index = index % self.length
    #     return super(DupDataset, self).get_img_info(true_index)
