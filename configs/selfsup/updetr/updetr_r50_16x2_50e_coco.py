'''
Author: your name
Date: 2022-01-02 21:20:55
LastEditTime: 2022-03-29 11:35:08
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/detr_r50_16x2_50e_coco.py
'''
_base_ = [
    '../_base_/datasets/updetr_detection.py', 
    '../_base_/updetr_default_runtime.py',
    '../_base_/schedules/updetr_schedule_1x.py', 
    '../_base_/models/updetr_r50_16x2_50e.py'
]