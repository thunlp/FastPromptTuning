#export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=3

DATE_TIME=$(date '+%y-%m-%d-%H:%M')

MODEL_SIZE=${1}
TASK_NAME=${2}
DATA_PATH=${3}
TUNE_METHOD=prompt
TARGET_LENGTH=${4}
SEED=${5}

SUFFIX=${MODEL_SIZE}-${TASK_NAME}-${TUNE_METHOD}

SAMPLE_SIZE=1000
# PRETRAIN_MODEL="google/t5-${MODEL_SIZE}-lm-adapt" # optional: set the path of pretrained language model
# TOKENIZER="google/t5-${MODEL_SIZE}-lm-adapt"
PRETRAIN_MODEL="/data1/private/huangyufei/pretrained_models/t5-${MODEL_SIZE}-lm-adapt"
TOKENIZER="/data1/private/huangyufei/pretrained_models/t5-${MODEL_SIZE}-lm-adapt"
SCORE_FILE="scores/${TASK_NAME}-${MODEL_SIZE}.csv"
mkdir -p scores
attribution_method=abs_activations


MODEL_ARGS="--model-name-or-path ${PRETRAIN_MODEL} \
            --tokenizer-name-or-path ${TOKENIZER}"
TRAIN_ARGS="--tune-method ${TUNE_METHOD} \
            --task ${TASK_NAME} \
            --datapath ${DATA_PATH} \
            --max-seq-length 492 \
            --max-target-length ${TARGET_LENGTH} \
            --early-stop -1 \
            --train-iters 30000 \
            --micro-batch-size 2 \
            --eval-micro-batch-size 64 \
            --global-batch-size 32 \
            --weight-decay 1e-5 \
            --seed ${SEED}"
LR_ARGS="--lr 0.3 \
         --lr-warmup-fraction 0.0 \
         --lr-decay-style constant"
PROMPT_ARGS="--prompt-length 20"
INTERGRATED_GRADIENTS_ARGS="--sample-size ${SAMPLE_SIZE} \
                            --attribution-method ${attribution_method} \
                            --score-file ${SCORE_FILE}"
python src/score_neuron.py $MODEL_ARGS $TRAIN_ARGS $LR_ARGS $PROMPT_ARGS \
                                          $INTERGRATED_GRADIENTS_ARGS