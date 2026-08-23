#!/usr/bin/env bash
# 零训练文本通道探针：4 变体 × 2 档 cfg_scale，8 卡各一个。
# 与训练同卡，必须设 --vram-limit-gib，理由见 legacy/run_infer8_film.sh。
# 注意 cfg_scale>1 会同时跑正负两侧，单步开销翻倍。
set -euo pipefail
VRAM=${VRAM:-30}
COMBO=(
  "none 1.0"  "still 1.0"  "left 1.0"  "right 1.0"
  "none 5.0"  "still 5.0"  "left 5.0"  "right 5.0"
)
for g in "$@"; do
  set -- ${COMBO[$g]}
  nohup python3 code/abot/probe_text_channel.py \
    --variant "$1" --cfg-scale "$2" \
    --device "cuda:$g" --vram-limit-gib "$VRAM" --allow-busy-gpu \
    --run-name probe_text --overwrite \
    > "logs/probe_text/gpu${g}_$1_cfg$2.log" 2>&1 &
  echo "gpu$g -> variant=$1 cfg=$2  pid=$!"
done
