#! /bin/bash
# Fine-tune FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH=/u/scr/yuhuiz/develop/factCC/modeling # absolute path to modeling directory
export DATA_PATH=/u/scr/yuhuiz/develop/factCC/pregenerated_data/generated_data/falke-claim-only # absolute path to data directory
export OUTPUT_PATH=/u/scr/yuhuiz/develop/factCC/pretrained_models/factcc-falke-claim-only # absolute path to model checkpoint

export TASK_NAME=factcc_generated
export MODEL_NAME=bert-base-uncased

python $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_lower_case \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --data_dir $DATA_PATH \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT_PATH/$MODEL_NAME-$TASK_NAME-finetune-$RANDOM/
