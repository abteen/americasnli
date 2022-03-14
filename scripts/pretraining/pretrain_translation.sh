#!/bin/bash

seed=${2}
lang=${1}

python run_pretraining.py \
  --collator_config configs/collator/pretraining/mlm_collator_default.yaml \
  --dataset_config configs/dataset/pretraining/linebyline_translated_dataset.yaml \
  --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
  --model_config configs/model/pretraining/pretraining_default.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  --training_args configs/training_arguments/pretraining/pretraining_default.yaml \
  experiment_name=mlm_${lang}_${seed} \
  seed=${seed} \
  use_wandb=True \
  log_directory=/projects/abeb4417/americasnli/logs/mlm-translated-1.0/ \
  output_directory=/rc_scratch/abeb4417/americasnli/mlm-translated-1.0/ \
  target_language=${lang}
