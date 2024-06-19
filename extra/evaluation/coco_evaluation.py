# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import os
import numpy as np
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json, _evaluate_predictions_on_coco

class MultiSourceCOCOEvaluator(COCOEvaluator):

    def __init__(self, *args, source_idx, index_diff, valid_idx_range, **kwargs):
        super(MultiSourceCOCOEvaluator, self).__init__(*args, **kwargs)
        self.source_idx = source_idx
        self.index_diff = index_diff
        self.valid_idx_range = valid_idx_range
        

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """        
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id # id2idx
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()} # idx2id
            for result in coco_results:
                category_id = result["category_id"] # index corresponding to the classifier head
                
                # if the predictions come from classifier for different sources, ignore
                if category_id < self.valid_idx_range[0] or category_id >= self.valid_idx_range[1]:
                    continue
                # modify class index with a given gap
                category_id -= self.index_diff
                
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                    logger = self._logger,
                    output_dir = self._output_dir,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res