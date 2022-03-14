#!/bin/bash


seed=${1}

python run_finetuning.py \
  --collator_config configs/collator/finetuning/collator_default.yaml \
  --dataset_config configs/dataset/finetuning/anli_es.yaml \
  --experiment_config configs/experiment/finetuning/nli_default.yaml \
  --model_config configs/model/finetuning/xlmr_large_pretrained_seq.yaml \
  --tokenizer_config configs/tokenizer/xlmr_large_tokenizer.yaml \
  --training_args configs/training_arguments/finetuning/zs_es_large.yaml \
  --trainer_config configs/trainer/default.yaml \
  experiment_name=xlmr_large_zero_shot_es_${seed} \
  log_directory=/projects/abeb4417/americasnli/logs/xlmr_large/ \
  output_directory=/rc_scratch/abeb4417/americasnli/xlmr_large/ \
  seed=${seed} \
  use_wandb=True \
  n_gpu=2 \
  visible_devices="0,1"

