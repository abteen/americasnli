#!/bin/bash

for lang in aym bzd cni gn hch nah oto quy shp tar
do
python run_pretraining.py \
  --collator_config configs/collator/pretraining/mlm_collator_default.yaml \
  --dataset_config configs/dataset/pretraining/linebyline_dataset.yaml \
  --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
  --model_config configs/model/pretraining/pretraining_default.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  --training_args configs/training_arguments/pretraining/pretraining_default.yaml \
  experiment_name=mlm \
  target_language=${lang} \
  seed=42

done