#! /bin/bash
# Train FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH=/u/scr/yuhuiz/develop/factCC/modeling # absolute path to modeling directory
export DATA_PATH=/u/scr/yuhuiz/develop/factCC/pregenerated_data/generated_data/data-pointers # absolute path to data directory
export OUTPUT_PATH=/u/scr/yuhuiz/develop/factCC/pretrained_models/factccx-cnndm-train # absolute path to model checkpoint


export TASK_NAME=factcc_generated
export MODEL_NAME=bert-base-uncased

python $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_from_scratch \
  --data_dir $DATA_PATH \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --overwrite_cache \
  --output_dir $OUTPUT_DIR/$MODEL_NAME-$TASK_NAME-train-$RANDOM/
