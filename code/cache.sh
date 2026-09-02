#!/usr/bin/env bash
# Cache the MiniMax-H3 inputs used by the released H3-World training run.
set -euo pipefail

cd "$(dirname "$0")/.."
source env.sh >/dev/null 2>&1 || true

REPO=DiffSynth-Studio-h3-v2
DATA=${DATA:-$PWD/data/clips}
META=${META:-$PWD/data/abot_meta_train_7872.jsonl}
CACHE=${CACHE:-$PWD/output/minimax_h3_abot/7872-cache}
GPUS=${CUDA_VISIBLE_DEVICES:-0}
NUM_GPUS=$(($(grep -o "," <<<"$GPUS," | wc -l)))

[ -d "$REPO" ] || { echo "Missing $REPO. Complete Setup first." >&2; exit 1; }
[ -f "$META" ] || { echo "Missing $META. Build and split the ABot clips first." >&2; exit 1; }

ACCELERATE_ARGS=()
if [ "$NUM_GPUS" -gt 1 ]; then
  ACCELERATE_ARGS=(--multi_gpu --num_processes "$NUM_GPUS" --num_machines 1)
fi

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch "${ACCELERATE_ARGS[@]}" \
  examples/minimax_h3/model_training/train_v2.py \
  --dataset_base_path "$DATA" \
  --dataset_metadata_path "$META" \
  --data_file_keys video,input_audio \
  --extra_inputs input_audio,input_image \
  --height 480 --width 832 --num_frames 124 \
  --dataset_repeat 1 --dataset_num_workers 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --output_path "$CACHE" \
  --silent_on_missing_audio \
  --task sft:data_process
