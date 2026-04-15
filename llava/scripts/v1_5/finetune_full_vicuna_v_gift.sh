#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LLAVA_CKPT_DIR="${REPO_ROOT}/llava/checkpoints"
DATASETS_DIR="${REPO_ROOT}/datasets"

BASE_OUTPUT_DIR="${REPO_ROOT}/checkpoints/llava-v1.5-7b"
RUN_NAME="${CUSTOM_NAME:-finetune_full_vicuna_v_gift}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

deepspeed llava/train/train_mem.py \
    --deepspeed "${SCRIPT_DIR}/../zero3.json" \
    --model_name_or_path "${LLAVA_CKPT_DIR}/vicuna-7b-v1.5" \
    --vision_tower       "${LLAVA_CKPT_DIR}/clip-vit-large-patch14-336" \
    --version v1 \
    --data_path "${DATASETS_DIR}/llava_v1_5_v_gift.json" \
    --image_folder "${DATASETS_DIR}" \
    --pretrain_mm_mlp_adapter "${LLAVA_CKPT_DIR}/llava-v1.5-7b-pretrain/mm_projector.bin" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}"
