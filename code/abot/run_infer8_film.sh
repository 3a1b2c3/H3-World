#!/usr/bin/env bash
# 与训练同卡并行推理最新 FiLM checkpoint。
# 训练每卡占 86 GB / 143.7 GB，所以必须给权重设常驻上限，让 DiffSynth 从 CPU 流式换入，
# 否则 vram_limit 默认吃满整张卡会把训练进程 OOM 掉。
set -euo pipefail
CKPT=${CKPT:-output/minimax_h3_abot/7872_film/step-3000.safetensors}
TAG=${TAG:-step3000_film}
VRAM=${VRAM:-30}
IDS=(
  a3ad9c24bda131dfa0ea18efe44a4e8b
  ba62b0242ba84e26341ba007cb24eb5a
  2e0dad2ecbd1cc7bad1f615da25a583f
  9460b04ef359848a6acdb483edcb6350
  230a5bf138a03b5a4a92bfcdf494a1db
  306458e18f73b6540d369452af1b8203
  7efdf07796748d868edda0c6d2fce53a
  d0b768c69b0f689131f4e7ffbc7a7384
)
for g in "$@"; do
  nohup python3 code/abot/infer_abot.py \
    --checkpoint "$CKPT" \
    --sample-id "${IDS[$g]}" \
    --device "cuda:$g" \
    --vram-limit-gib "$VRAM" \
    --allow-busy-gpu \
    --run-name "${TAG}_gpu${g}" \
    --overwrite \
    > "logs/infer8_film/gpu${g}.log" 2>&1 &
  echo "gpu$g -> ${IDS[$g]}  pid=$!"
done
