#!/bin/bash

for seed in 925 271
do
python run_finetuning.py \
  --collator_config configs/collator/finetuning/collator_default.yaml \
  --dataset_config configs/dataset/finetuning/anli_es.yaml \
  --experiment_config configs/experiment/finetuning/nli_default.yaml \
  --model_config configs/model/finetuning/xlmr_pretrained_seq.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  --training_args configs/training_arguments/finetuning/zs_es.yaml \
  --trainer_config configs/trainer/default.yaml \
  experiment_name=zero_shot_es_${seed} \
  seed=${seed} \
  use_wandb=True

done
