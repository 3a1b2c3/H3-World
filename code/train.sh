#!/usr/bin/env bash
# Train the per-latent text-injection LoRA on the directed attention mask.
# This is the current, released configuration: 7872 training clips, rank-32
# LoRA on qkv_proj/out_proj, 20 epochs.
#
# Two things must both point at the v2 framework fork, or training silently
# uses a different attention mask against the same data:
#   - this script's `cd "$REPO"` into DiffSynth-Studio-h3-v2
#   - train_v2.py itself, which strips the editable-install finder before
#     importing diffsynth (see "Installation" in README.md for why that
#     finder is a problem)
#
# Usage:
#   bash code/train.sh                    # 4 GPUs (0-3), defaults below
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash code/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."               # repo root
source env.sh >/dev/null 2>&1 || true

REPO=DiffSynth-Studio-h3-v2
CACHE=${CACHE:-$PWD/output/minimax_h3_abot/7872-cache}
OUT=${OUT:-$PWD/output/minimax_h3_abot/7872_directed}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=$(($(grep -o "," <<<"$GPUS,"  | wc -l)))

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --multi_gpu --num_processes "$NUM_GPUS" \
  --num_machines 1 --main_process_port 29751 \
  examples/minimax_h3/model_training/train_v2.py \
  --dataset_base_path "$CACHE" \
  --data_file_keys video,input_audio \
  --extra_inputs input_audio,input_image \
  --height 480 --width 832 --num_frames 124 \
  --dataset_repeat 1 --dataset_num_workers 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "$OUT" \
  --lora_base_model dit --lora_target_modules qkv_proj,out_proj --lora_rank 32 \
  --use_gradient_checkpointing --silent_on_missing_audio \
  --enable_tensorboard_log --save_steps 2000 \
  --task sft:train
