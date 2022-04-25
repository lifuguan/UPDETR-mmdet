'''
Author: your name
Date: 2022-04-11 19:55:33
LastEditTime: 2022-04-11 22:32:34
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /up-detr/coco_10.py
'''

import numpy as np
from pycocotools.coco import COCO
import random
from tqdm import tqdm
import json
import glob
import os

# Store annotations and train2014/val2014/... in this folder
dataDir = '/disk1/lihao/data/coco'

# ./COCO/annotations/instances_train2014.json
annFile = '{}/annotations/instances_train2017.json'.format(dataDir)

coco = COCO(annFile)

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

all_ann_ids = []
for cls in CLASSES:
    # Get ID number of this class
    cls_id = coco.getCatIds(catNms=[cls])
    ann_ids = coco.getAnnIds(catIds=cls_id)
    all_ann_ids.extend(random.sample(ann_ids, len(ann_ids) // 10))

ann_json = []
for key in tqdm(all_ann_ids):
    ann_json.append(coco.anns[key])


img_ids = []
for per_ann_ids in all_ann_ids:
    img_ids.append(coco.anns[per_ann_ids]['image_id'])
unique_img_ids = np.unique(img_ids)
img_json = []
for key in tqdm(unique_img_ids):
    img_json.append(coco.imgs[key])

all_json = {
    "images": img_json,
    "annotations": ann_json,
    "categories": [coco.cats[ids] for ids in coco.cats]
}

instances_train2017 = json.dumps(all_json)
f = open(os.path.join(dataDir+'/annotations/10percent_train2017.json'), 'w')
f.write(instances_train2017)
f.close()