#!/bin/bash
FILE=glue.py
MODEL_TYPE=bert
MODEL_PATH=bertwiki103 #bertwiki103  littlebert  reformer16k  reformer512 earlybert
REFORMER_HASHES=0
GPU_SIZE=32
GPU=1
GLUE_DIR=./data/glue

for LR in "1e-5" "2e-5" "5e-5" "1e-4"
do
    for TASK_NAME in CoLA MNLI MRPC QNLI QQP RTE SNLI SST-2 STS-B WNLI 
    do
        OUT_DIR=./val_output_models/$TASK_NAME-$MODEL_PATH-lr$LR-epoch$EPOCHS-hashes$REFORMER_HASHES
        echo $OUT_DIR

        CUDA_VISIBLE_DEVICES=$GPU \
            ~/.pyenv/shims/python $FILE \
            --model_type $MODEL_TYPE \
            --model_name_or_path input_models/$MODEL_PATH \
            --task_name $TASK_NAME \
            --do_train \
            --do_eval \
            --do_lower_case \
            --reformer_hashes $REFORMER_HASHES \
            --data_dir $GLUE_DIR/$TASK_NAME \
            --per_gpu_train_batch_size $GPU_SIZE \
            --learning_rate $LR \
            --output_dir $OUT_DIR
    done
done




