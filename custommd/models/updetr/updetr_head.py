# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, multi_apply)
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.detr_head import DETRHead


@HEADS.register_module()
class UPDETRHead(DETRHead):
    def __init__(self, 
                *args,
                num_classes,
                feature_recon=False,  # TODO disable feature reconstruction
                query_shuffle=False, 
                mask_ratio=0.1, 
                num_patches=10,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    bg_cls_weight=0.1,
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=1.0), **kwargs): 
        # NOTE inherit changes `__class__`
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is UPDETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        super(UPDETRHead, self).__init__(*args, num_classes=num_classes, loss_cls=loss_cls, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature
        # align the patch feature dim to query patch dim.
        self.patch2query = nn.Linear(self.in_channels, self.embed_dims)
        
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.feature_recon = feature_recon
        if self.feature_recon:
            # align the transformer feature to the CNN feature, which is used for the feature reconstruction
            self.feature_align = MLP(self.embed_dims, self.embed_dims, self.in_channels, 2)
        self.query_shuffle = query_shuffle
        assert self.num_query % num_patches == 0  # for simplicity
        query_per_patch = self.num_query // num_patches
        # the attention mask is fixed during the pre-training
        self.attention_mask = torch.ones(self.num_query, self.num_query) * float('-inf')
        for i in range(query_per_patch):
            self.attention_mask[i * query_per_patch:(i + 1) * query_per_patch,
            i * query_per_patch:(i + 1) * query_per_patch] = 0

    def forward(self, feats, patches, img_metas, training=False):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            patches (list[Tensor]): Random patches from images. NOTE
            img_metas (list[dict]): List of image information.
            training (bool): indicate training mode or testing mode NOTE


        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, patches, img_metas_list, training=training)

    def forward_single(self, x, patch_feats, img_metas, training=False):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.
            training (bool): indicate training mode or testing mode NOTE

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """


        patch_feature_gt = self.avgpool(patch_feats).flatten(1)

        # align the dim of patch feature with object query with a linear layer
        # pay attention to the difference between "torch.repeat" and "torch.repeat_interleave"
        # it converts the input from "1,2,3,4" to "1,2,3,4,1,2,3,4,1,2,3,4" by torch.repeat
        # "1,2,3,4" to "1,1,1,2,2,2,3,3,3,4,4,4" by torch.repeat_interleave, which is our target.
        patch_feats = self.patch2query(patch_feature_gt) \
            .view(img_metas[0]["bs"], img_metas[0]["batch_num_patches"], -1) \
            .repeat_interleave(self.num_query // self.num_patches, dim=1) \
            .permute(1, 0, 2) \
            .contiguous()

        # if object query shuffle, we shuffle the index of object query embedding,
        # which simulate to adding patch feature to object query randomly.
        idx = torch.randperm(self.num_query) if self.query_shuffle else torch.arange(self.num_query)

        # NOTE query 
        if training is True:
            # for training, it uses fixed number of query patches.
            mask_query_patch = (torch.rand(self.num_query, img_metas[0]["bs"], 1, device=patch_feats.device) > self.mask_ratio).float()
            # NOTE mask some query patch and add query embedding
            patch_feature = patch_feats * mask_query_patch + self.query_embedding.weight[idx, :].unsqueeze(1).repeat(1, img_metas[0]["bs"], 1)
        else:
            patch_feature = patch_feats + self.query_embedding.weight[:self.num_query, :].unsqueeze(1).repeat(1, img_metas[0]["bs"], 1)
            

        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        
        # NOTE Transformer
        if training is True:
            # param: Query, Key, Value, pos_embed, attn_masks
            # outs_dec: [nb_dec, bs, num_query, embed_dim]
            outs_dec, _ = self.transformer(x, masks, patch_feature, pos_embed#, attn_masks=self.attention_mask.to(patch_feature.device)
            )
        else:
            outs_dec, _ = self.transformer(x, masks, patch_feature, pos_embed#, attn_masks=self.attention_mask.to(patch_feature.device)[:self.num_query,:self.num_query]
            )


        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(
            self.activate(self.reg_ffn(outs_dec))).sigmoid()
        if self.feature_recon:   # TODO feature reconstruction is now disable
            all_feat_preds = self.feature_align(outs_dec)
            return all_cls_scores, all_bbox_preds, all_feat_preds
        else:
            return all_cls_scores, all_bbox_preds

    # over-write because patches are needed as inputs for Transformer.
    def forward_train(self,
                      x,
                      patches,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            patches (list[Tensor]): Random patches from images. NOTE
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, patches, img_metas, training = True) # calling forward() func
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def simple_test_bboxes(self, feats, patches, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            patches (list[Tensor]): Random patches from images. NOTE
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, patches, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    """
    TODO feature loss needs to be adapted
    """
    # @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list', 'all_feat_preds_list'))
    # def loss(self,
    #          all_cls_scores_list,
    #          all_bbox_preds_list,
    #          all_feat_preds_list,
    #          gt_bboxes_list,
    #          gt_labels_list,
    #          img_metas,
    #          gt_bboxes_ignore=None):
    #     """"Loss function.

    #     Only outputs from the last feature level are used for computing
    #     losses by default.

    #     Args:
    #         all_cls_scores_list (list[Tensor]): Classification outputs
    #             for each feature level. Each is a 4D-tensor with shape
    #             [nb_dec, bs, num_query, cls_out_channels].
    #         all_bbox_preds_list (list[Tensor]): Sigmoid regression
    #             outputs for each feature level. Each is a 4D-tensor with
    #             normalized coordinate format (cx, cy, w, h) and shape
    #             [nb_dec, bs, num_query, 4].
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
    #             with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels_list (list[Tensor]): Ground truth class indices for each
    #             image with shape (num_gts, ).
    #         img_metas (list[dict]): List of image meta information.
    #         gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
    #             which can be ignored for each image. Default None.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     # NOTE defaultly only the outputs from the last feature scale is used.
    #     all_cls_scores = all_cls_scores_list[-1]
    #     all_bbox_preds = all_bbox_preds_list[-1]
    #     all_feat_preds = all_feat_preds_list[-1] # for loss feature(UP-DETR)
    #     assert gt_bboxes_ignore is None, \
    #         'Only supports for gt_bboxes_ignore setting to None.'

    #     num_dec_layers = len(all_cls_scores)
    #     all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
    #     all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
    #     all_gt_bboxes_ignore_list = [
    #         gt_bboxes_ignore for _ in range(num_dec_layers)
    #     ]
    #     img_metas_list = [img_metas for _ in range(num_dec_layers)]

    #     losses_cls, losses_bbox, losses_iou, losses_feat = multi_apply(
    #         self.loss_single, all_cls_scores, all_bbox_preds, all_feat_preds,
    #         all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
    #         all_gt_bboxes_ignore_list)

    #     loss_dict = dict()
    #     # loss from the last decoder layer
    #     loss_dict['loss_cls'] = losses_cls[-1]
    #     loss_dict['loss_bbox'] = losses_bbox[-1]
    #     loss_dict['loss_iou'] = losses_iou[-1]
    #     loss_dict['loss_feat'] = losses_feat[-1] # for loss feature(UP-DETR)
    #     # loss from other decoder layers
    #     num_dec_layer = 0
    #     for loss_cls_i, loss_bbox_i, loss_iou_i, losses_feat_i in zip(losses_cls[:-1],
    #                                                    losses_bbox[:-1],
    #                                                    losses_iou[:-1],
    #                                                    losses_feat[:-1]):
    #         loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
    #         loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
    #         loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
    #         loss_dict[f'd{num_dec_layer}.losses_feat'] = losses_feat_i
    #         num_dec_layer += 1
    #     return loss_dict

    # def loss_single(self,
    #                 cls_scores,
    #                 bbox_preds,
    #                 feat_preds,
    #                 gt_bboxes_list,
    #                 gt_labels_list,
    #                 img_metas,
    #                 gt_bboxes_ignore_list=None):
    #     """"Loss function for outputs from a single decoder layer of a single
    #     feature level.

    #     Args:
    #         cls_scores (Tensor): Box score logits from a single decoder layer
    #             for all images. Shape [bs, num_query, cls_out_channels].
    #         bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
    #             for all images, with normalized coordinate (cx, cy, w, h) and
    #             shape [bs, num_query, 4].
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
    #             with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels_list (list[Tensor]): Ground truth class indices for each
    #             image with shape (num_gts, ).
    #         img_metas (list[dict]): List of image meta information.
    #         gt_bboxes_ignore_list (list[Tensor], optional): Bounding
    #             boxes which can be ignored for each image. Default None.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components for outputs from
    #             a single decoder layer.
    #     """
    #     num_imgs = cls_scores.size(0)
    #     cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    #     bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    #     feat_preds_list = [feat_preds[i] for i in range(num_imgs)]
    #     cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
    #                                        gt_bboxes_list, gt_labels_list,
    #                                        img_metas, gt_bboxes_ignore_list)
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets
    #     labels = torch.cat(labels_list, 0)
    #     label_weights = torch.cat(label_weights_list, 0)
    #     bbox_targets = torch.cat(bbox_targets_list, 0)
    #     bbox_weights = torch.cat(bbox_weights_list, 0)

    #     # classification loss
    #     cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    #     # construct weighted avg_factor to match with the official DETR repo
    #     cls_avg_factor = num_total_pos * 1.0 + \
    #         num_total_neg * self.bg_cls_weight
    #     if self.sync_cls_avg_factor:
    #         cls_avg_factor = reduce_mean(
    #             cls_scores.new_tensor([cls_avg_factor]))
    #     cls_avg_factor = max(cls_avg_factor, 1)

    #     loss_cls = self.loss_cls(
    #         cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    #     # Compute the average number of gt boxes across all gpus, for
    #     # normalization purposes
    #     num_total_pos = loss_cls.new_tensor([num_total_pos])
    #     num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

    #     # construct factors used for rescale bboxes
    #     factors = []
    #     for img_meta, bbox_pred in zip(img_metas, bbox_preds):
    #         img_h, img_w, _ = img_meta['img_shape']
    #         factor = bbox_pred.new_tensor([img_w, img_h, img_w,
    #                                        img_h]).unsqueeze(0).repeat(
    #                                            bbox_pred.size(0), 1)
    #         factors.append(factor)
    #     factors = torch.cat(factors, 0)

    #     # DETR regress the relative position of boxes (cxcywh) in the image,
    #     # thus the learning target is normalized by the image size. So here
    #     # we need to re-scale them for calculating IoU loss
    #     bbox_preds = bbox_preds.reshape(-1, 4)
    #     bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
    #     bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

    #     # regression IoU loss, defaultly GIoU loss
    #     loss_iou = self.loss_iou(
    #         bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

    #     # regression L1 loss
    #     loss_bbox = self.loss_bbox(
    #         bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
    #     return loss_cls, loss_bbox, loss_iou



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x