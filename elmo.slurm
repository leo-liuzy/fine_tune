#!/bin/bash

## Job Name

#SBATCH --job-name=squad-elmo

#SBATCH --partition=stf-gpu

#SBATCH --account=stf

#SBATCH --gres=gpu:P100:1


#SBATCH --time=10:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.

## For mox, --mem may be more than 100G depending on the memory of your nodes.

## For ikt, --mem may be 58G or more depending on the memory of your nodes.

## See above section on "Specifying memory" for choices for --mem.

#SBATCH --mem=100G

## Specify the working directory for this job

#SBATCH --chdir=/gscratch/stf/zeyuliu2/fine_tune

##turn on e-mail notification

#SBATCH --mail-type=ALL

#SBATCH --mail-user=zeyuliu2@cs.washington.edu

## export all your environment variables to the batch job session

#SBATCH --export=all

export PROJ_DIR="/gscratch/stf/zeyuliu2/fine_tune"
export SQUAD_DIR="$PROJ_DIR/data/squad1_1"
export HOME="/usr/lusers/zeyuliu2"
export BATCH_SIZE=8
export ACC_STEP=1
export MODEL_NAME="bert-base-uncased"
export MODEL_PATH="$PROJ_DIR/cache/bert-base-uncased-pytorch_model.bin"
export CONFIG_PATH="$PROJ_DIR/cache/bert-base-uncased-config.json"
export TOKENIZER_PATH="$PROJ_DIR/cache/bert-base-uncased-vocab.txt"


# bert as extractor
python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --elmo_style --output_dir out/fine_tune_top_layer+elmo --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# bert fine_tune
python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --elmo_style --output_dir out/fine_tune+elmo --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP

# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_NAME --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --output_dir out/fine_tune_elmo_style --overwrite_output_dir  --save_steps 22090 --elmo_style --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
