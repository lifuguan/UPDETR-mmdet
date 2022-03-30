'''
Author: your name
Date: 2022-03-03 11:08:20
LastEditTime: 2022-03-30 00:24:00
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/_base_/models/updetr_r50_16x2_50e.py
'''


custom_imports = dict(
    imports=[
        'custommd.models.detectors.updetr',
        'custommd.models.updetr.updetr_head',
        'mmdet.models',
        'custommd.datasets.selfdet',
        'custommd.pipelines.transforms',
        'custommd.models.updetr.transformer'
    ],
    allow_failed_imports=False)
model = dict(
    type='UPDETR',
    backbone=dict(
        type='ResNet',        
        depth=50,
        num_stages=4,
        out_indices=(3, ), # 只输出最后一层
        frozen_stages=4,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar')),
    bbox_head=dict(
        type='UPDETRHead',
        num_classes=4,
        num_patches=10,
        in_channels=2048,
        transformer=dict(
            type='CustomTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))
