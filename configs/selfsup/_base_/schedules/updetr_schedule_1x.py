'''
Author: your name
Date: 2022-03-29 10:03:59
LastEditTime: 2022-03-29 20:35:36
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/_base_/schedules/selfsup_schedule_1x.py
'''

# optimizer
# this is different from the original 1x schedule that use SGD
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.0001)

optimizer_config = dict(
    # detect_anomalous_params=True, # 查找未使用的参数（可注释）
    grad_clip=dict(max_norm=0.1, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=60)
# evaluation = dict(save_best='mAP')