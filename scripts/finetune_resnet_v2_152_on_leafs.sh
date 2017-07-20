#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the leafs dataset
# 2. Fine-tunes a ResNetV2-152 model on the leafs training set.
# 3. Evaluates the model on the leafs validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_resnet_v2_152_on_leafs.sh
set -e

# Where the pre-trained ResNetV2-152 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/leafs-models/resnet_v2_152

# Where the dataset is saved to.
DATASET_DIR=/tmp/leafs

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_152.ckpt ]; then
  wget http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
  tar -xvf resnet_v2_152_2017_04_14.tar.gz
  mv resnet_v2_152.ckpt ${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_152.ckpt
  rm resnet_v2_152_2017_04_14.tar.gz
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=leafs \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 6000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=leafs \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v2_152 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_152.ckpt \
  --checkpoint_exclude_scopes=resnet_v2_152/logits \
  --trainable_scopes=resnet_v2_152/logits \
  --max_number_of_steps=6000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=leafs \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v2_152

# Fine-tune all the new layers for 3000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=leafs \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --model_name=resnet_v2_152 \
  --max_number_of_steps=3000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=leafs \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v2_152
