#!/bin/bash

export CUDA_VISIBLE_DEVICES=3,4,5,6
GPU_NUM=4
SEED=2021
DATA_PATH=data/mnli
TASK=mnli
TARGET_LENGTH=10

# model size is LARGE
MODEL_SIZE=large
TRAIN_ITERS=30000
MODEL_PARALLEL_SIZE=1
PER_GPU_BATCH_SIZE=4
# vanilla prompt train
bash scripts/prompt-train.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} ${GPU_NUM}
# layer dropping
PROGRESSIVE_TRAIN_ITERS=30001
WITH_DECODER=0
for PROGRESSIVE_LAYERS in 6 12 18 ;
do
    bash scripts/layer-dropping-partial.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} \
                                        ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} ${PROGRESSIVE_TRAIN_ITERS} \
                                        ${PROGRESSIVE_LAYERS} ${WITH_DECODER} ${GPU_NUM}
done
# ffn reduction
for PROGRESSIVE_FFN_DIMENSTIONS in 704 1408 2112 ;
do
    bash scripts/ffn-reduction-partial.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} \
                                        ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} ${PROGRESSIVE_TRAIN_ITERS} \
                                        ${PROGRESSIVE_FFN_DIMENSTIONS} ${WITH_DECODER} ${GPU_NUM}
done
# compound reduction
for i in "6 704" "12 1408" "18 2112" ;
do
    set -- $i
    PROGRESSIVE_LAYERS=$1
    PROGRESSIVE_FFN_DIMENSTIONS=$2
    bash scripts/compound-reduction-partial.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} \
                                               ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} ${PROGRESSIVE_TRAIN_ITERS} \
                                               ${PROGRESSIVE_LAYERS} ${PROGRESSIVE_FFN_DIMENSTIONS} ${WITH_DECODER} ${GPU_NUM}
done

