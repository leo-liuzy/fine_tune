#!/bin/bash
export SQUAD_DIR="/homes/iws/zeyuliu2/fine_tune/data/squad1_1"
export BATCH_SIZE=4
export ACC_STEP=2

# bert as extractor
python runs/run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --output_dir out/fine_tune_top_layer --overwrite_output_dir  --save_steps 22090 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# python runs/run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --output_dir out/fine_tune_elmo_style --overwrite_output_dir  --save_steps 22090 --elmo_style --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP

# pkill -KILL -u zeyuliu2
