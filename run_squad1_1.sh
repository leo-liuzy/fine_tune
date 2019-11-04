#!/bin/bash
export SQUAD_DIR="/data2/zeyuliu2/fine_tune/data/squad1_1"

python runs/run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrain --output_dir out/fine_tune_top_layer --overwrite_output_dir  --save_steps 5522
python runs/run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --output_dir out/fine_tune  --overwrite_output_dir --save_steps 5522