#!/usr/bin/env bash
# 分块续写：一段接一段，用上一段的**尾帧**当下一段的首帧。
#
# 为什么这条路不用改架构：推理时首帧是通过 `keyframes=[PIL图片]` 传的，
# 收的就是普通图片，所以"喂上一段的尾帧"是现成能力。
#
# 每段各给各的动作文本（这里用随机按键，seed 逐段递增），所以按键控制照样有效。
# 已知代价：段间只靠一帧衔接，运动的速度和方向是断的；且第 N 段基于生成帧，
# 误差会累积。这个脚本就是拿来量这两件事有多严重的。
#
# 用法:
#   bash code/abot/run_chunked_continue.sh 3           # 3 段，约 15.5s
#   CHUNKS=5 DEVICE=cuda:1 bash code/abot/run_chunked_continue.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=${CKPT:-output/minimax_h3_abot/7872_text/step-9840.safetensors}
SID=${SID:-a3ad9c24bda131dfa0ea18efe44a4e8b}
TAG=${TAG:-chunked_step9840}
DEVICE=${DEVICE:-cuda:1}
VRAM=${VRAM:-30}
CHUNKS=${1:-${CHUNKS:-3}}
OUTDIR=output/abot_inference/${TAG}

mkdir -p logs/longvid "$OUTDIR"
prev_frame=""
for ((c = 0; c < CHUNKS; c++)); do
  echo "── 第 $((c+1))/$CHUNKS 段 ──"
  extra=()
  [ -n "$prev_frame" ] && extra=(--first-frame "$prev_frame")
  python3 code/abot/infer_abot.py \
    --checkpoint "$CKPT" --sample-id "$SID" \
    --device "$DEVICE" \
    --action-random "$c" \
    --vram-limit-gib "$VRAM" --allow-busy-gpu \
    --run-name "${TAG}_c${c}" --overwrite \
    "${extra[@]}" \
    > "logs/longvid/chunk${c}.log" 2>&1

  gen=$(ls "output/abot_inference/${TAG}_c${c}"/*/generated.mp4 | head -1)
  [ -n "$gen" ] || { echo "  ✗ 第 $((c+1)) 段没有产物" >&2; exit 1; }
  prev_frame="$OUTDIR/tail_c${c}.png"
  # 取尾帧留给下一段。用 python 而不是 ffmpeg -sseof：后者对短片定位不稳。
  python3 - "$gen" "$prev_frame" <<'PY'
import sys, av
last = None
with av.open(sys.argv[1]) as ct:
    for f in ct.decode(video=0):
        last = f
last.to_image().convert("RGB").save(sys.argv[2])
print(f"  尾帧 -> {sys.argv[2]}")
PY
done

echo
echo "拼接 $CHUNKS 段 …"
: > "$OUTDIR/list.txt"
for ((c = 0; c < CHUNKS; c++)); do
  echo "file '$(realpath "$(ls "output/abot_inference/${TAG}_c${c}"/*/generated.mp4 | head -1)")'" \
    >> "$OUTDIR/list.txt"
done
FF=$(realpath ../envs/minimax_h3/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2)
"$FF" -y -loglevel error -f concat -safe 0 -i "$OUTDIR/list.txt" -c copy "$OUTDIR/joined.mp4"
echo "✓ $OUTDIR/joined.mp4  ($(python3 -c "
import av,sys
with av.open('$OUTDIR/joined.mp4') as c:
    n=sum(1 for _ in c.decode(video=0)); print(f'{n} 帧 = {n/24:.1f}s')"))"
