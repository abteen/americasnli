#!/bin/bash

lang=${1}
seed=${2}

python run_pretraining.py \
  --collator_config configs/collator/pretraining/mlm_collator_default.yaml \
  --dataset_config configs/dataset/pretraining/linebyline_dataset.yaml \
  --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
  --model_config configs/model/pretraining/pretraining_xlmr_large.yaml \
  --tokenizer_config configs/tokenizer/xlmr_large_tokenizer.yaml \
  --training_args configs/training_arguments/pretraining/pretraining_xlmr_large_ta.yaml \
  experiment_name=${lang} \
  log_directory=/projects/abeb4417/americasnli/logs/xlmr_large/mlm/ \
  output_directory=/rc_scratch/abeb4417/americasnli/xlmr_large/mlm/ \
  target_language=${lang} \
  seed=42
