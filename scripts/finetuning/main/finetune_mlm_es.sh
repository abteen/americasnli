#!/bin/bash

seed=${2}
lang=${1}

python run_finetuning.py \
    --collator_config configs/collator/finetuning/collator_default.yaml \
    --dataset_config configs/dataset/finetuning/anli_es.yaml \
    --experiment_config configs/experiment/finetuning/nli_default.yaml \
    --model_config configs/model/finetuning/xlmr_pretrained_seq.yaml \
    --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
    --training_args configs/training_arguments/finetuning/zs_es.yaml \
    --trainer_config configs/trainer/default.yaml \
    experiment_name=zero_shot_mlm_es_${lang}_${seed} \
    seed=${seed} \
    use_wandb=True \
    model_settings.init.pretrained_model_name_or_path=/rc_scratch/abeb4417/americasnli/mlm/${lang}/final_model/ \
    log_directory=/projects/abeb4417/americasnli/logs/mlm_es/ \
    output_directory=/rc_scratch/abeb4417/americasnli/mlm_es/