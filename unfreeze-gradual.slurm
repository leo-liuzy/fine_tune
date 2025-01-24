#!/bin/bash

## Job Name

#SBATCH --job-name=[50%][6-11]

#SBATCH --partition=stf-gpu

#SBATCH --account=stf

#SBATCH --gres=gpu:P100:1


#SBATCH --time=9:00:00

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

export PROJ_DIR="/home/leo/fine_tune"
export SQUAD_DIR="$PROJ_DIR/data/squad1_1"
export LOG_DIR="$PROJ_DIR/logs"
export HOME="/gscratch/stf/zeyuliu2"
export EPOCH=4
export LR=0.00005
export RANGE=None
export BERT_LAYER_RANGE=[6-11]
export LAYERNORM_RANGE=None
export BATCH_SIZE=10
export ACC_STEP=1
export MODEL_NAME="bert-base-uncased"
export MODEL_PATH="$PROJ_DIR/cache/bert-base-uncased-pytorch_model.bin"
export CONFIG_PATH="$PROJ_DIR/cache/bert-base-uncased-config.json"
export TOKENIZER_PATH="$PROJ_DIR/cache/bert-base-uncased-vocab.txt"
# export TBD_DIR="/gscratch/stf/zeyuliu2/fine_tune/out/fine_tune_top_layer+unfreeze8/checkpoint-11080"
# tbd
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --apply_adapter --bottleneck_size 64 --unfreeze_top_k_bert_layer 0 --output_dir $PROJ_DIR/out/fine_tune_top_layer+adapter64 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# echo "normal-adapter-lr$LR-bs36-epoch$EPOCH-adapter64-adapterActivaion-$K_LAYERNORM"LayerNorm.txt
python runs/run_squad_unfreeze_gradual.py --run_mode single_run --learning_rate $LR --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file "$SQUAD_DIR/train-v1.1.json" --predict_file "$SQUAD_DIR/dev-v1.1.json" --adapter_range $RANGE --bottleneck_size 64 --adapter_activation 1 --unfreeze_bert_layer_range $BERT_LAYER_RANGE --unfreeze_layernorm_range $LAYERNORM_RANGE --output_dir "$PROJ_DIR/out" --logging_dir "$LOG_DIR" --overwrite_output_dir --save_steps 7386 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP --num_train_epochs $EPOCH --gpu_id 0  > "unfreeze-gradual-lr$LR-bs36-epoch$EPOCH-adapter64-range$RANGE-adapterActivaion-unfreeze$LAYERNORM_RANGE"LayerNorm"-unfreeze$BERT_LAYER_RANGE"BertLayer.txt 2>&1

# bert as extractor
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --unfreeze_top_k_layer 1 --output_dir out/fine_tune_top_layer+unfreeze1 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --unfreeze_top_k_layer 2 --output_dir out/fine_tune_top_layer+unfreeze2 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --unfreeze_top_k_layer 4 --output_dir out/fine_tune_top_layer+unfreeze4 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --freeze_pretrained --unfreeze_top_k_layer 8 --output_dir out/fine_tune_top_layer+unfreeze8 --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP



# bert fine_tune
# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_PATH --config_name $CONFIG_PATH --tokenizer_name $TOKENIZER_PATH --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --output_dir out/fine_tune --overwrite_output_dir  --save_steps 5540 --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP

# python runs/run_squad.py --model_type bert --model_name_or_path $MODEL_NAME --do_train --do_eval --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --output_dir out/fine_tune_elmo_style --overwrite_output_dir  --save_steps 22090 --elmo_style --per_gpu_train_batch_size $BATCH_SIZE --per_gpu_eval_batch_size $BATCH_SIZE --gradient_accumulation_steps $ACC_STEP

