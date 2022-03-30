'''
Author: lifuguan
Date: 2022-03-18 17:49:55
Based: 
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/custommd/datasets/selfdet.py
'''

import numpy as np

import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose

from mmselfsup.datasets.base import BaseDataset
from mmselfsup.datasets.builder import DATASETS, PIPELINES, build_datasource

from custommd.pipelines.transforms import CustomCompose

import time 

def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
    x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
    patch = img.crop((x, y, x + sw, y + sh))
    return patch, x, y, sw, sh

@DATASETS.register_module()
class SelfDetDataset(BaseDataset):
    """The dataset outputs multiple views of an image.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MultiViewDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = MultiViewDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self, data_source, backbone_pipeline, query_pipeline, format_pipeline, prefetch=False, num_patches=10):

        self.data_source = build_datasource(data_source)
        
        detection_transform = [
            build_from_cfg(p, PIPELINES) for p in backbone_pipeline
        ]
        query_transform = [
            build_from_cfg(p, PIPELINES) for p in query_pipeline
        ]
        format_transform = [
            build_from_cfg(p, PIPELINES) for p in format_pipeline
        ]
        self.detection_transform = CustomCompose(detection_transform) # 从backbone输入
        self.format_transform = Compose(format_transform) # 对backbone输入的数据转换成datacontainer
        self.query_transform = Compose(query_transform) # 从query输入

        self.prefetch = prefetch
        self.num_patches = num_patches


    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        w, h = img.size
        if w<=16 or h<=16:
            return self[(idx+1)%len(self)]
        # the format of the dataset is same with COCO.
        target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
        labels = []
        boxes = []
        patches = []
        while len(labels) < self.num_patches:
            patch, x, y, sw, sh = get_random_patch_from_img(img)
            boxes.append([x, y, x + sw, y + sh])
            labels.append(1)
            patches.append(self.query_transform(patch))
        target['labels'] = torch.tensor(labels)
        target['boxes'] = torch.tensor(boxes)
        img, target = self.detection_transform(img, target)

        results = self.format_transform(dict(img=img, gt_bboxes=target['boxes'], gt_labels=target['labels']))

        return dict(
            img_metas=idx, # img_metas不能为空，保证损失函数可以计算
            img=results['img'], gt_bboxes=results['gt_bboxes'], gt_labels=results['gt_labels'], patches=torch.stack(patches, dim=0)
        )
        
    def evaluate(self, results, logger=None):
        return NotImplemented

