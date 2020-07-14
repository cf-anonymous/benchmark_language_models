#!/bin/bash
FILE=glue.py
MODEL_TYPE=reformer16k
MODEL_PATH=reformer16k #bertwiki103  littlebert  reformer16k  reformer512
REFORMER_HASHES=4
GPU_SIZE=32
GPU=1
GLUE_DIR=./data/glue

for TASK_NAME in MNLI MRPC CoLA RTE SNLI STS-B WNLI SST-2 QNLI QQP 
do
    for LR in "1e-4" "5e-5" "2e-5"  "1e-5"  
    do
        OUT_DIR=./val_output_models/$TASK_NAME-$MODEL_PATH-lr$LR-hashes$REFORMER_HASHES
        echo $OUT_DIR

        CUDA_VISIBLE_DEVICES=$GPU \
            ~/.pyenv/shims/python $FILE \
            --model_type $MODEL_TYPE \
            --model_name_or_path input_models/$MODEL_PATH \
            --task_name $TASK_NAME \
            --do_train \
            --do_eval \
            --do_lower_case \
            --max_seq_length 256 \
            --reformer_hashes $REFORMER_HASHES \
            --data_dir $GLUE_DIR/$TASK_NAME \
            --per_gpu_train_batch_size $GPU_SIZE \
            --learning_rate $LR \
            --output_dir $OUT_DIR
    done
done




