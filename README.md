<!--
 * @Author: your name
 * @Date: 2021-12-13 18:14:32
 * @LastEditTime: 2022-03-30 11:50:43
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /research_workspace/README.md
-->
# UP-DETR : mmdetection(2.19.0) implementation

This project hosts the code for implementing the UP-DETR algorithms based on the [**official project**](https://github.com/dddzg/up-detr). Due to the inheritance designation, it enables developers as well as researchers to integrate into their projects more easily and elegantly.

该UP-DETR项目是基于[**官方项目**](https://github.com/dddzg/up-detr)并在 **mmdetection(2.19.0)** 上实现的。由于采用继承设计，它使得开发人员和研究人员能够更轻松，更优雅地集成到自己的项目中。

## TODO
`Feature reconstruction`  is not intergrated into this repo.

## Installation
It requires the following OpenMMLab packages:
- **MMDetection >= 2.14.0**
- Linux
- Python 3.7
- PyTorch >= 1.6
- torchvision 0.7.0
- CUDA 10.1
- NCCL 2
- GCC >= 5.4.0
- MMCV >= 1.3.8

## Usage
### Semi-Supervised Object Detection on COCO 10% labeled data
Extract 10% labeld data from each categories in COCO `instance_train2017.json`.
```python 
python ./coco_10.py
```
### Data preparation

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). The data structure looks like below:

```bash
data/
├── coco
│   ├── annotations
│   │   ├── panoptic_{train,val}2017.json
│   │   ├── instance_{train,val}2017.json
│   │   ├── panoptic_{train,val}2017/  # panoptic png annotations
│   │   ├── image_info_test-dev2017.json  # for test-dev submissions
│   ├── train2017
│   ├── val2017
│   ├── test2017
├── imagenet  # for unsupervised learning
│   ├── annotations
│   │   ├── train.txt
│   │   ├── val.txt
│   ├── train2017
│   ├── val2017
```
### symbolic路径
For the convenience of files managment, I strongly recommend to use symbolic link to point to another files with huge storge memories.
```bash
ln -s /disk1/lihao/data data
ln -s /disk1/lihao/work_dirs work_dirs
```
### Training and testing

You can run training and testing without slurm by directly using mim for instance segmentation like below:

```bash
# UP-DETR training
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH  
./tools_selfsup/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

Example:
python ./tools_selfsup/train.py configs/selfsup/updetr/updetr_r50_16x2_50e_coco.py --gpus 1

# UP-DETR inference
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  --show --out  ${OUTPUT_FILE} --eval segm

Example:
./tools/dist_test.sh configs/det/solo/UP-DETR_r50_fpn_coco.py work_dirs/UP-DETR_r50_fpn_coco/latest.pth 1 --eval segm
```

## VSCode debugger launch config 
```json 
{
    "env": {"PYTHONPATH" : "${workspaceRoot}"},
    "name": "knet:debug",
    "type": "python",
    "request": "launch",
    "program": "${workspaceRoot}/tools_selfsup/train.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": ["configs/selfsup/updetr/updetr_r50_16x2_50e_coco.py","--gpus", "1"]
},
```
