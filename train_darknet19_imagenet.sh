#!/bin/bash

DATASET_DIR=/mnt/imagenetdisk/tf_imagenet
TRAIN_DIR=/home/wenxichen/dl_yolo2/logs/darknet19_imagenet
python imagenet_train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --train_image_size=224