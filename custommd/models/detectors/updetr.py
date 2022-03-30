'''
Author: your name
Date: 2022-03-03 10:50:51
LastEditTime: 2022-03-29 19:50:09
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/custommd/models/detectors/updetr.py
'''

from mmselfsup.models.builder import ALGORITHMS
from mmdet.models.detectors import DETR
import time 

@ALGORITHMS.register_module()
class UPDETR(DETR):
    '''
    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.
    '''
    def __init__(self, *args, **kwargs):
        super(UPDETR, self).__init__(*args, **kwargs)
        self.bs = 32 # NOTE 在修改img_per_gpu时，该值需要同步修改
        
    # over-write `extract_feat` because:
    # patches are also required during backbone forward
    def extract_feat(self, img, patch):

        """Directly extract features from the backbone+neck."""
        x_feats = self.backbone(img)  # 没有FPN，直接用最后一层layer的输出
        patch_feats = self.backbone(patch) # 没有FPN，直接用最后一层layer的输出

        if self.with_neck:
            x_feats = self.neck(x_feats)
            patch_feats = self.neck(patch_feats)
        return x_feats, patch_feats

    # over-write `forward_train` because:
    # 1. patches are also required in pretraining procedure
    # 2. img_metas need to be generated right here
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, patches):
        # time1=time.time()
        batch_size, _, height, width = img.shape
        train_img_metas = [
            dict(
                batch_input_shape=(height, width),
                bs = self.bs,  # patches information
                batch_num_patches = patches.shape[1], # patches information
                img_shape=(height, width, 3)) for _ in range(batch_size)
                
        ]
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in train_img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        
        patches = patches.flatten(0, 1)
        x_feats, patch_feats = self.extract_feat(img, patches)
        losses = self.bbox_head.forward_train(x_feats, patch_feats,  train_img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None)
        # print("after forward", time.time()-time1)
        return losses

    # over-write `forward_train` because:
    # 1. patches are also required in pretraining procedure
    # 2. img_metas need to be generated right here
    def simple_test(self, img, patches, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_size, _, height, width = img.shape
        test_img_metas = [
            dict(
                batch_input_shape=(height, width),               
                bs = self.bs,  # patches information
                batch_num_patches = patches.shape[1], # patches information
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in test_img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        
        patches = patches.flatten(0, 1)

        x_feats, patch_feats = self.extract_feat(img, patches)
        results_list = self.bbox_head.simple_test(
            x_feats, patch_feats, test_img_metas, rescale=rescale)
        bbox_results = [
            self.bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

