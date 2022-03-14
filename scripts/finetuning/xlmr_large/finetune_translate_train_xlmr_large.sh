
seed=${2}
lang=${1}

  python run_finetuning.py \
    --collator_config configs/collator/finetuning/collator_default.yaml \
    --dataset_config configs/dataset/finetuning/anli_translate_train.yaml \
    --experiment_config configs/experiment/finetuning/nli_default.yaml \
    --model_config configs/model/finetuning/xlmr_large_pretrained_seq.yaml \
    --tokenizer_config configs/tokenizer/xlmr_large_tokenizer.yaml \
    --training_args configs/training_arguments/finetuning/translate_train_xlmr_large.yaml \
    --trainer_config configs/trainer/default.yaml \
    experiment_name=xlmr_large_translate_train_${lang}_${seed} \
    seed=${seed} \
    use_wandb=True \
    log_directory=/projects/abeb4417/americasnli/logs/xlmr_large/translate_train/ \
    output_directory=/rc_scratch/abeb4417/americasnli/xlmr_large/translate_train/ \
    training_arguments.metric_for_best_model=eval_translate_train_${lang}_accuracy \
    dataset_settings.eval_dataset.language=${lang} \
    dataset_settings.train_dataset.language=${lang} \
    n_gpu=2 \
    visible_devices="0,1"

