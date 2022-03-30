'''
Author: your name
Date: 2022-03-18 15:21:57
LastEditTime: 2022-03-29 17:14:33
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/_base_/datasets/selfsup_detection.py
'''

data_source = 'ImageNet'
dataset_type = 'SelfDetDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]

train_backbone_pipeline = [
    dict(type='RandomResize',sizes=scales, max_size=600),
    dict(type='CustomToTensor'),
    dict(type='CustomNormalize', **img_norm_cfg),
]
test_backbone_pipeline = [
    dict(type='RandomResize',sizes=[480], max_size=600),
    dict(type='CustomToTensor'),
    dict(type='CustomNormalize', **img_norm_cfg),
]

train_query_pipeline = [
    dict(type='Resize', size=(128, 128)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_query_pipeline = [
    dict(type='Resize', size=(128, 128)),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
format_pipeline = [
    dict(type='CustomDefaultFormatBundle')
]

# prefetch
prefetch = False

# dataset summary
data = dict(
    imgs_per_gpu=32,  # need to modify `bs` on updetr.py  
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train2017',
            ann_file=None,
        ),
        backbone_pipeline=train_backbone_pipeline,
        query_pipeline=train_query_pipeline,
        format_pipeline=format_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/val2017',
            ann_file='data/imagenet/annotations/val.txt',
        ),
        backbone_pipeline=test_backbone_pipeline,
        format_pipeline=format_pipeline,
        query_pipeline=test_query_pipeline,
        prefetch=prefetch))