#!/bin/bash

## Job Name

#SBATCH --job-name=ada_between

#SBATCH --partition=stf-gpu

#SBATCH --account=stf

#SBATCH --gres=gpu:P100:1


#SBATCH --time=24:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.

## For mox, --mem may be more than 100G depending on the memory of your nodes.

## For ikt, --mem may be 58G or more depending on the memory of your nodes.

## See above section on "Specifying memory" for choices for --mem.

#SBATCH --mem=100G

## Specify the working directory for this job

#SBATCH --chdir=/gscratch/stf/zeyuliu2/fine_tune

##turn on e-mail notification

#SBATCH --mail-type=ALL

#SBATCH --mail-user=zeyuliu2@uw.edu

## export all your environment variables to the batch job session

#SBATCH --export=all

export PROJ_DIR="/gscratch/stf/zeyuliu2/fine_tune"
export SQUAD_DIR="$PROJ_DIR/data/squad1_1"
export LOG_DIR="$PROJ_DIR/logs"
export HOME="/gscratch/stf/zeyuliu2"
export BATCH_SIZE=12
export ACC_STEP=1
export MODEL_NAME="bert-base-uncased"
export MODEL_PATH="$PROJ_DIR/cache/bert-base-uncased-pytorch_model.bin"
export CONFIG_PATH="$PROJ_DIR/cache/bert-base-uncased-config.json"
export TOKENIZER_PATH="$PROJ_DIR/cache/bert-base-uncased-vocab.txt"
# export TBD_DIR="/gscratch/stf/zeyuliu2/fine_tune/out/fine_tune_top_layer+unfreeze8/checkpoint-11080"
# tbd
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --apply_adapter --bottleneck_size 64 --unfreeze_top_k_bert_layer 0 --output_dir $PROJ_DIR/out/fine_tune_top_layer+adapter64 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
python runs/run_squad.py --run_mode "single_run" --learning_rate 0.0001 --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --overwrite_output_dir --train_file "$SQUAD_DIR/train-v1.1.json" --predict_file "$SQUAD_DIR/dev-v1.1.json" --apply_adapter_between_layer --bottleneck_size 64 --adapter_activation 0 --unfreeze_top_k_bert_layer 0 --output_dir "$PROJ_DIR/out" --logging_dir "$LOG_DIR" --save_steps 7386 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP --num_train_epochs 10 --gpu_id 0  > normal-adapter-lr0.0001-bs12-epoch10-adapter64-adapterNoActivation-AdaBetweenLayer.txt 2>&1

# bert as extractor
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --unfreeze_top_k_layer 1 --output_dir out/fine_tune_top_layer+unfreeze1 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --unfreeze_top_k_layer 2 --output_dir out/fine_tune_top_layer+unfreeze2 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --unfreeze_top_k_layer 4 --output_dir out/fine_tune_top_layer+unfreeze4 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --unfreeze_top_k_layer 8 --output_dir out/fine_tune_top_layer+unfreeze8 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP



# bert fine_tune
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --output_dir out/fine_tune --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP

# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_NAME --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --output_dir out/fine_tune_elmo_style --overwrite_output_dir  --save_steps 22090 --elmo_style --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP

