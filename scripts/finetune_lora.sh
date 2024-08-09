#!/bin/bash


PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"

image_folder=/path/to/vision-flan-images/
data_path=/path/to/vision-flan-json/

lora_r=$1
lora_alpha=$(($lora_r * 2))

deepspeed --master_port=2997 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path "lmsys/vicuna-7b-v1.3" \
    --version $PROMPT_VERSION \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/ft_vision-flan_lora_r-${lora_r} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1792 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --lora_r $lora_r \
    --report_to tensorboard \
    --lora_alpha $lora_alpha \
