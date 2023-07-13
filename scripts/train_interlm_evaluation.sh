#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model openllama_evaluation \
    --model_path internlm/internlm-7b\
    --data_path ./generate_data/train.json\
    --max_length 2048\
    --save_path  ./ckpt/openllama_evaluation/\
    --log_path ./ckpt/openllama_evaluation/rest


