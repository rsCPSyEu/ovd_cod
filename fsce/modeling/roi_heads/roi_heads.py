# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
# from detectron2.utils.registry import Registry
# from detectron2.modeling.roi_heads import StandardROIHeads

# from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.box_regression import Box2BoxTransform

# from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
# from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
# from detectron2.modeling.roi_heads.mask_head import build_mask_head

from .fast_rcnn import (
    FastRCNNContrastOutputs,
    CosineSimOutputLayers
)
from ..contrastive_loss import (
    SupConLoss,
    SupConLossV2,
    ContrastiveHead,
)
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["ContrastiveROIHeads",]


class _ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(_ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]  # post_nms_top_k proposals have no matche will be drop here
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            # use ground truth bboxes as super-high quality proposals for training
            # with logits = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # matched_idxs in [0, M)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            iou, _ = match_quality_matrix.max(dim=0)
            # random sample batche_size_per_image proposals with positive fraction
            # NOTE: only matched proposals will be returned
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.iou = iou[sampled_idxs]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
        # proposals_with_gt, List[Instances], fields = ['gt_boxes', 'gt_classes', ‘proposal_boxes’, 'objectness_logits']

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()

        
class _StandardROIHeads(_ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(_StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.output_layer_name = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        # self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer_name)(
        #     cfg, self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        # )
        if self.output_layer_name == 'CosineSimOutputLayers':
            self.box_predictor = CosineSimOutputLayers(cfg, self.box_head._output_size, self.num_classes, self.cls_agnostic_bbox_reg)
        else:
            raise Exception('Unknown output layer name: {}'.format(self.output_layer_name))


    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
            proposals (List[Instance]): fields=[proposal_boxes, objectness_logits]
                post_nms_top_k proposals for each image， len = N

            targets (List[Instance]):   fields=[gt_boxes, gt_classes]
                gt_instances for each image, len = N
        """
        del images
        if self.training:
            # label and sample 256 from post_nms_top_k each images
            # has field [proposal_boxes, objectness_logits ,gt_classes, gt_boxes]
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            # FastRCNNOutputs.losses()
            # {'loss_cls':, 'loss_box_reg':}
            losses = self._forward_box(features_list, proposals)  # get losses from fast_rcnn.py::FastRCNNOutputs
            return proposals, losses  # return to rcnn.py line 201
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  # [None, 256, POOLER_RESOLU, POOLER_RESOLU]
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class ContrastiveROIHeads(_StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: on
        self.fc_dim               = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self.contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.num_classes          = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.loss_version         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION
        self.contrast_iou_thres   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        self.reweight_func        = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC

        self.cl_head_only         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: off

        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
        if self.loss_version == 'V1':
            self.criterion = SupConLoss(self.temperature, self.contrast_iou_thres, self.reweight_func)
        elif self.loss_version == 'V2':
            self.criterion = SupConLossV2(self.temperature, self.contrast_iou_thres)
        self.criterion.num_classes = self.num_classes  # to be used in protype version

    def _forward_box(self, features, proposals):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        box_features_contrast = self.encoder(box_features)
        del box_features

        if self.weight_decay:
            storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate

        outputs = FastRCNNContrastOutputs(
            self.box2box_transform,
            pred_class_logits, # [N, K+1]
            pred_proposal_deltas, # [N, K*4]
            proposals,
            self.smooth_l1_beta, # 0.0
            box_features_contrast, # [N, 128]
            self.criterion,
            self.contrast_loss_weight,
            self.box_reg_weight,
            self.cl_head_only,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances