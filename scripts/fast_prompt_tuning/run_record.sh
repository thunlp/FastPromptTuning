#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
GPU_NUM=4
SEED=2022
DATA_PATH=data/record
TASK=record
TARGET_LENGTH=200

# model size is LARGE
MODEL_SIZE=large
TRAIN_ITERS=30000
MODEL_PARALLEL_SIZE=1
PER_GPU_BATCH_SIZE=4
# vanilla prompt train
bash scripts/prompt-train.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} ${GPU_NUM}
# layer dropping
PROGRESSIVE_TRAIN_ITERS=6000,6000,6000
PROGRESSIVE_LAYERS=6,12,18
WITH_DECODER=1
bash scripts/layer-dropping.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} \
                               ${PROGRESSIVE_TRAIN_ITERS} ${PROGRESSIVE_LAYERS} ${WITH_DECODER} ${GPU_NUM}
# ffn reduction
## first get scores of each neuron
bash scripts/score-neuron.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${SEED}
PROGRESSIVE_FFN_DIMENSTIONS=704,1408,2112
bash scripts/ffn-reduction.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} \
                              ${PROGRESSIVE_TRAIN_ITERS} ${PROGRESSIVE_FFN_DIMENSTIONS} ${WITH_DECODER} ${GPU_NUM}
# compound reduction
bash scripts/compound-reduction.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} \
                                   ${PROGRESSIVE_TRAIN_ITERS}  ${PROGRESSIVE_LAYERS} ${PROGRESSIVE_FFN_DIMENSTIONS} ${WITH_DECODER} ${GPU_NUM}

# model size is XL
MODEL_SIZE=xl
TRAIN_ITERS=15000
MODEL_PARALLEL_SIZE=2
PER_GPU_BATCH_SIZE=2
# vanilla prompt train
bash scripts/prompt-train.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} ${GPU_NUM}
# layer dropping
PROGRESSIVE_TRAIN_ITERS=3000,3000,3000
PROGRESSIVE_LAYERS=6,12,18
WITH_DECODER=1
bash scripts/layer-dropping.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} \
                               ${PROGRESSIVE_TRAIN_ITERS} ${PROGRESSIVE_LAYERS} ${WITH_DECODER} ${GPU_NUM}
# ffn reduction
## first get scores of each neuron
bash scripts/score-neuron.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${SEED}
PROGRESSIVE_FFN_DIMENSTIONS=1280,2560,3840
bash scripts/ffn-reduction.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} \
                              ${PROGRESSIVE_TRAIN_ITERS} ${PROGRESSIVE_FFN_DIMENSTIONS} ${WITH_DECODER} ${GPU_NUM}
# compound reduction
bash scripts/compound-reduction.sh ${MODEL_SIZE} ${TASK} ${DATA_PATH} ${TARGET_LENGTH} ${PER_GPU_BATCH_SIZE} ${SEED} ${TRAIN_ITERS} ${MODEL_PARALLEL_SIZE} \
                                   ${PROGRESSIVE_TRAIN_ITERS}  ${PROGRESSIVE_LAYERS} ${PROGRESSIVE_FFN_DIMENSTIONS} ${WITH_DECODER} ${GPU_NUM}