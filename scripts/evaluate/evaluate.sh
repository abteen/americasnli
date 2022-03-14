#!/bin/bash

#### ZERO SHOT ENGLISH EVALUATION ####

#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_en_271/final_model/ \
#  --experiment_name zero_shot_en_271 \
#  --langs xnli \
#  --log_dir logs/evaluation/zero_shot/
#
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_en_271/final_model/ \
#  --experiment_name zero_shot_en_271 \
#  --langs anli \
#  --log_dir logs/evaluation/zero_shot/

#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_en_925/final_model/ \
#  --experiment_name zero_shot_en_925 \
#  --langs xnli \
#  --log_dir logs/evaluation/zero_shot/
#
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_en_925/final_model/ \
#  --experiment_name zero_shot_en_925 \
#  --langs anli \
#  --log_dir logs/evaluation/zero_shot/


#### ZERO SHOT SPANISH EVALUATION ####

#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_es_271/final_model/ \
#  --experiment_name zero_shot_es_271 \
#  --langs anli \
#  --log_dir logs/evaluation/zero_shot/
#
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_es_925/final_model/ \
#  --experiment_name zero_shot_es_925 \
#  --langs anli \
#  --log_dir logs/evaluation/zero_shot/

#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_es_271/final_model/ \
#  --experiment_name zero_shot_es_xnli_271 \
#  --langs xnli \
#  --log_dir logs/evaluation/zero_shot/
#
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_es_925/final_model/ \
#  --experiment_name zero_shot_es_xnli_925 \
#  --langs xnli \
#  --log_dir logs/evaluation/zero_shot/

#### MLM (es) EVALUATION ####

#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/mlm_es/zero_shot_mlm_es_${lang}_925/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/mlm_es_925/
#done

#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/mlm_es_271/zero_shot_mlm_es_${lang}_271/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/mlm_es_271/
#done

#### MLM (en) EVALUATION ####

#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/mlm_en/zero_shot_mlm_en_${lang}_925/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/mlm_en_925/
#done

#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/mlm_en/zero_shot_mlm_en_${lang}_271/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/mlm_en_271/
#done

#### TRANSLATE-TEST ####
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_es_271/final_model/ \
#  --test_file data/translate_test/translate_test.test.tsv \
#  --test_format translate-train \
#  --log_dir logs/evaluation/translate_test/ \
#  --experiment_name translate_test_271 \
#  --langs anli

#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/zero_shot/zero_shot_es_925/final_model/ \
#  --test_file data/translate_test/translate_test.test.tsv \
#  --test_format translate-train \
#  --log_dir logs/evaluation/translate_test/ \
#  --experiment_name translate_test_925 \
#  --langs anli


#### TRANSLATE-TRAIN ####
#for lang in shp tar #aym bzd cni gn hch nah oto quy
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/translate_train/translate_train_${lang}_271/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/translate_train_271/
#done

#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/translate_train/translate_train_${lang}_925/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/translate_train_925/
#done

#TRANSLATE-TRAIN MLM ###

#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/translate_train_mlm/translate_train_mlm_${lang}_925/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/translate_train_mlm_925/
# done

#### MLM-AUG EN ####

#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/mlm-aug_en/zero_shot_mlm-aug_en_${lang}_42/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/mlm-aug-en-42/
#done
#
##### MLM-AUG EN ####
#
#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/mlm-aug_es/zero_shot_mlm-aug_es_${lang}_42/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/mlm-aug-es-42/
#done

#for lang in aym bzd cni gn hch nah oto quy shp tar
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/translate_train_mlm-aug/translate_train_mlm-aug_${lang}_42/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/mlm-aug-translate-train/
#done


#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/xlmr_large/xlmr_large_zero_shot_en_42/final_model/ \
#  --experiment_name xlmr_large_zero_shot_en_42_anli \
#  --langs anli \
#  --log_dir logs/evaluation/other_models/xlmr_large/

#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/xlmr_large/xlmr_large_zero_shot_es_42/final_model/ \
#  --experiment_name xlmr_large_zero_shot_es_42_anli \
#  --langs anli \
#  --log_dir logs/evaluation/other_models/xlmr_large/

#for lang in bzd
#do
#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/xlmr_large/translate_train/xlmr_large_translate_train_bzd_42/final_model/ \
#  --experiment_name ${lang} \
#  --langs ${lang} \
#  --log_dir logs/evaluation/other_models/xlmr_large/translate_train/
#done

for lang in bzd
do
python evaluate_xnli.py \
  --load_from_path /rc_scratch/abeb4417/americasnli/xlmr_large/mlm_es/xlmr_large_zero_shot_mlm_es_${lang}_42/final_model/ \
  --experiment_name ${lang} \
  --langs ${lang} \
  --log_dir logs/evaluation/other_models/xlmr_large/mlm_es/
done

#XLMR Large Translate Test

#python evaluate_xnli.py \
#  --load_from_path /rc_scratch/abeb4417/americasnli/xlmr_large/xlmr_large_zero_shot_es_42/final_model/ \
#  --test_file data/translate_test/translate_test.test.tsv \
#  --test_format translate-train \
#  --log_dir logs/evaluation/other_models/xlmr_large/translate_test/ \
#  --experiment_name translate_test_42 \
#  --langs anli