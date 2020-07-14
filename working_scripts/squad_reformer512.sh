#!/bin/bash
FILE=squad.py
MODEL_TYPE=reformer512
MODEL_PATH=reformer512 #bertwiki103  littlebert  reformer16k  reformer512
REFORMER_HASHES=8
GPU_SIZE=32
GPU=7
echo $OUT_DIR

for LR in "5e-5" "1e-4" "1e-5" "2e-5"
do
    for TASK_NAME in v2.0 
    do
        OUT_DIR=./val_output_models/squad$TASK_NAME-$MODEL_PATH-lr$LR-hashes$REFORMER_HASHES
        echo $OUT_DIR

        CUDA_VISIBLE_DEVICES=$GPU \
            ~/.pyenv/shims/python $FILE \
            --model_type $MODEL_TYPE \
            --model_name_or_path input_models/$MODEL_PATH \
            --do_eval \
            --do_train \
            --do_lower_case \
            --reformer_hashes $REFORMER_HASHES \
            --data_dir ./data/squad$TASK_NAME \
            --predict_file dev-$TASK_NAME.json \
            --train_file train-$TASK_NAME.json \
            --per_gpu_train_batch_size $GPU_SIZE \
            --learning_rate $LR \
            --output_dir $OUT_DIR
    done
done

#done


#FILE=squad.py
#MODEL_TYPE=bert
#MODEL_PATH=bert-base-uncased #bertwiki103  littlebert  reformer16k  reformer512
#EVAL_FILE=./data/squad/dev-v1.1.json
#REFORMER_HASHES=0
#GPU_SIZE=32
#GPU=0
#EPOCHS=3.0
#echo $OUT_DIR
#
#for TASK_NAME in v1.1 v2.0
#do
#    for LR in "1e-5" "2e-5" "5e-5" "1e-4"
#    do
#        OUT_DIR=./output_models/squad$TASK_NAME-$MODEL_PATH-lr$LR-epoch$EPOCHS-hashes$REFORMER_HASHES
#        echo $OUT_DIR
#
#        CUDA_VISIBLE_DEVICES=$GPU \
#            ~/.pyenv/shims/python $FILE \
#            --model_type $MODEL_TYPE \
#            --model_name_or_path $MODEL_PATH \
#            --do_eval \
#            --do_train \
#            --do_lower_case \
#            --reformer_hashes $REFORMER_HASHES \
#            --predict_file ./data/squad/dev-$TASK_NAME.json \
#            --train_file ./data/squad/train-$TASK_NAME.json \
#            --per_gpu_train_batch_size $GPU_SIZE \
#            --learning_rate $LR \
#            --num_train_epochs $EPOCHS \
#            --output_dir $OUT_DIR
#    done
#done



