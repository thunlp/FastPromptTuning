#!/bin/bash

export OMP_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=3
DATE_TIME=$(date '+%y-%m-%d-%H_%M')


GPU_NUM="${@: -1}"

MODEL_SIZE=${1}
TASK_NAME=${2}
DATA_PATH=${3}
TARGET_LENGTH=${4}
BATCH_SIZE=${5}
EVAL_BATCH_SIZE=$(expr ${BATCH_SIZE} \* 4)
SEED=${6}
TRAIN_ITERS=${7}
MODEL_PARALLEL_SIZE=${8}
PROGRESSIVE_TRAIN_ITERS=${9}
PROGRESSIVE_FFN_DIMENSTIONS=${10}
WITH_DECODER=${11}
FFN_SELECT_METHOD=score

TUNE_METHOD=prompt
DROPOUT=0.0

SUFFIX=${MODEL_SIZE}-${TASK_NAME}-${TUNE_METHOD}-fr-${PROGRESSIVE_FFN_DIMENSTIONS}dims
SUFFIX=${SUFFIX}-${SEED}

SAVE_PATH=./checkpoints/${TASK_NAME}/t5-${SUFFIX}
# PRETRAIN_MODEL="google/t5-${MODEL_SIZE}-lm-adapt" # optional: set the path of pretrained language model
# TOKENIZER="google/t5-${MODEL_SIZE}-lm-adapt"
PRETRAIN_MODEL="/data1/private/huangyufei/pretrained_models/t5-${MODEL_SIZE}-lm-adapt"
TOKENIZER="/data1/private/huangyufei/pretrained_models/t5-${MODEL_SIZE}-lm-adapt"
SCORE_FILE="scores/${TASK_NAME}-${MODEL_SIZE}.csv"

MODEL_ARGS="--model-name-or-path ${PRETRAIN_MODEL} \
            --tokenizer-name-or-path ${TOKENIZER}"
TRAIN_ARGS="--train \
            --tune-method ${TUNE_METHOD} \
            --task ${TASK_NAME} \
            --datapath ${DATA_PATH} \
            --max-seq-length 492 \
            --max-target-length ${TARGET_LENGTH} \
            --dropout ${DROPOUT} \
            --early-stop -1 \
            --train-iters ${TRAIN_ITERS} \
            --micro-batch-size ${BATCH_SIZE} \
            --eval-micro-batch-size ${EVAL_BATCH_SIZE} \
            --global-batch-size 32 \
            --weight-decay 1e-5 \
            --valid-interval 500 \
            --seed ${SEED} \
            --model-parallel-size ${MODEL_PARALLEL_SIZE}"
LOG_ARGS="--log-interval 10 --tensorboard-dir ${SAVE_PATH}/tensorboard"
LR_ARGS="--lr 0.3 \
         --lr-warmup-fraction 0.0 \
         --lr-decay-style constant"
SAVE_ARGS="--save ${SAVE_PATH}"
PROMPT_ARGS="--prompt-length 20"
FFN_PROGRESSIVE_ARGS="--encoder-ffn-progressive \
                     --progressive-ffn-dimentions ${PROGRESSIVE_FFN_DIMENSTIONS} \
                     --progressive-train-iters-ffn ${PROGRESSIVE_TRAIN_ITERS} \
                     --select-ffn-method ${FFN_SELECT_METHOD} \
                     --score-file ${SCORE_FILE}"
if [ ${WITH_DECODER} == 1 ]; then
  FFN_PROGRESSIVE_ARGS="${FFN_PROGRESSIVE_ARGS} --decoder-ffn-progressive"
fi
PROGRESSIVE_ARGS=${FFN_PROGRESSIVE_ARGS}

export WORLD_SIZE=$(expr ${GPU_NUM} / ${MODEL_PARALLEL_SIZE})
DATA_PARALLEL_SIZE=$(expr ${GPU_NUM} / ${MODEL_PARALLEL_SIZE})
DISTRIBUTED_CMD=""

if [ ${GPU_NUM} == 1 ]; then
  DISTRIBUTED_CMD=""
else
  DISTRIBUTED_CMD="-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=${DATA_PARALLEL_SIZE}"
fi
mkdir -p logs
mkdir -p logs/${TASK_NAME}
python ${DISTRIBUTED_CMD} src/train.py \
       $MODEL_ARGS $TRAIN_ARGS $LR_ARGS $SAVE_ARGS $LOG_ARGS $PROMPT_ARGS $PROGRESSIVE_ARGS \
       2>&1 | tee logs/${TASK_NAME}/log-${SUFFIX}-${DATE_TIME}.txt

cp logs/${TASK_NAME}/log-${SUFFIX}-${DATE_TIME}.txt ${SAVE_PATH}/log-${SUFFIX}-${DATE_TIME}.txt
