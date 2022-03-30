
###
 # @Author: your name
 # @Date: 2021-12-29 17:55:49
 # @LastEditTime: 2022-03-29 17:19:19
 # @LastEditors: Please set LastEditors
 # @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 # @FilePath: /research_workspace/train.sh
### 

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
# export PYTHONPATH=$PWD:$PYTHONPATH  
# ./tools_det/dist_train.sh configs/det/knet/mask_rcnn_r50_fpn_ctw1500_instance.py 7

# Mask RCNN
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  
# export PYTHONPATH=$PWD:$PYTHONPATH  
# ./tools/dist_train.sh configs/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py 7 

# Mask RCNN training
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=$PWD:$PYTHONPATH  
python ./tools_selfsup/train.py configs/selfsup/updetr/updetr_r50_16x2_50e_coco.py --gpus 1
# ./tools_det/dist_train.sh configs/det/knet/oursnet_s3_r50_fpn_1x_coco.py 7

# # Mask RCNN demo
# python ./tools_det/demo.py data/coco/train2017/000000175057.jpg work_dirs/mask_rcnn_r50_fpn_coco/bear_maskrcnn1.png configs/det/knet/mask_rcnn_r50_fpn_coco.py work_dirs/mask_rcnn_r50_fpn_coco/epoch_1.pth

# # K-Net training
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
# export PYTHONPATH=$PWD:$PYTHONPATH  
# ./tools_det/dist_train.sh configs/det/knet/detr_r50_16x2_50e_coco.py 7

# ./tools_det/dist_train.sh configs/det/knet/swin_t_p4_fpn_1x_coco.py 8
# ./tools_det/dist_train.sh configs/det/knet/swin_t_p4_fpn_1x_ctw1500_instance.py 8
# ./tools_det/dist_train.sh configs/det/knet/sparse_rcnn_r50_fpn_1x_coco.py 8
# ./tools_det/dist_train.sh configs/det/knet/decoupled_solo_r50_fpn_ctw1500_instance.py 8
# ./tools_det/dist_train.sh configs/det/knet/queryinst_r50_fpn_1x_coco.py 8
# ./tools_det/dist_train.sh configs/det/knet/deformable_detr_r50_16x2_50e_coco.py 8

# ./tools_det/dist_train.sh configs/det/knet/knet_s3_r50_fpn_1x_coco.py 7 # --load-from work_dirs/epoch_12.pth

# # K-Net demo
# python ./tools_det/demo.py data/coco/train2017/000000189095.jpg work_dirs/tmp/class_1_ins_2/demo_5.png configs/det/knet/knet_s3_r50_fpn_1x_coco.py work_dirs/knet_s3_r50_fpn_1x_coco_bear/epoch_5.pth

# python ./tools_det/demo.py data/coco/train2017/000000175057.jpg work_dirs/decoupled_solo_r50_fpn_coco/bear_solo.png configs/det/knet/decoupled_solo_r50_fpn_coco.py work_dirs/decoupled_solo_r50_fpn_coco/latest.pth


