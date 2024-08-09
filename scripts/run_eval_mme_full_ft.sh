PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"

model=$1
question_dir=$2 # /path/to/mixlora_data/mme/

python llava/cmoa_eval/evaluate_mme_pretrain.py \
    --lora_enable False \
    --model_name_or_path checkpoints/$model \
    --version $PROMPT_VERSION \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/eval_full_ft \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --question_dir $question_dir \