export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_SIZE=large

bash scripts/score-neuron.sh ${MODEL_SIZE} qqp data/qqp 10 2021
bash scripts/score-neuron.sh ${MODEL_SIZE} mnli data/mnli 10 2021
bash scripts/score-neuron.sh ${MODEL_SIZE} squad data/squad2 200 2021
bash scripts/score-neuron.sh ${MODEL_SIZE} record data/record 200 2021
bash scripts/score-neuron.sh ${MODEL_SIZE} xsum data/xsum 512 2021

MODEL_SIZE=xl

bash scripts/score-neuron.sh ${MODEL_SIZE} qqp data/qqp 10 2021
bash scripts/score-neuron.sh ${MODEL_SIZE} mnli data/mnli 10 2021
bash scripts/score-neuron.sh ${MODEL_SIZE} squad data/squad2 200 2021
bash scripts/score-neuron.sh ${MODEL_SIZE} record data/record 200 2021
bash scripts/score-neuron.sh ${MODEL_SIZE} xsum data/xsum 512 2021