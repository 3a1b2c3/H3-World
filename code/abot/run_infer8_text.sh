#!/usr/bin/env bash
# 8 卡并行推理逐 latent 文本注入的 checkpoint，**跑完自动出核对页**。
#
# 样本沿用 FiLM 那轮的同 8 条（legacy/run_infer8_film.sh），同 seed，方便横向对比。
# 训练已结束、8 卡全空时不需要限显存；若要和训练同卡挤着跑，设 VRAM=30。
#
# 用法:
#   bash code/abot/run_infer8_text.sh 0 1 2 3 4 5 6 7   # 8 卡全上，完事出页
#   bash code/abot/run_infer8_text.sh 3 5               # 只用 3、5 两张卡
#   CKPT=.../step-5000.safetensors TAG=step5000_text bash ... 0 1
#   CFG=5 bash ...          # 开 CFG（负 prompt 是动作零参考，放大的正是动作服从度）
#   VRAM=30 bash ...        # 和训练同卡挤着跑
#   NO_BUILD=1 bash ...     # 只推理，不出页
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=${CKPT:-output/minimax_h3_abot/7872_text/step-9840.safetensors}
TAG=${TAG:-step9840_text}
CFG=${CFG:-1.0}
VRAM=${VRAM:-}
NO_BUILD=${NO_BUILD:-0}
OUT_HTML=${OUT_HTML:-docs/action_prompt_text_viz.html}
ARTIFACT_HTML=${ARTIFACT_HTML:-.cache/artifact_text_viz.html}

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
[ "$#" -gt 0 ] || { echo "用法: bash $0 <gpu> [<gpu> ...]" >&2; exit 1; }

mkdir -p logs/infer8_text
VRAM_ARGS=()
[ -n "$VRAM" ] && VRAM_ARGS=(--vram-limit-gib "$VRAM" --allow-busy-gpu)

pids=(); gpus=()
for g in "$@"; do
  python3 code/abot/infer_abot.py \
    --checkpoint "$CKPT" \
    --sample-id "${IDS[$g]}" \
    --device "cuda:$g" \
    --cfg-scale "$CFG" \
    "${VRAM_ARGS[@]}" \
    --run-name "${TAG}_gpu${g}" \
    --overwrite \
    > "logs/infer8_text/gpu${g}.log" 2>&1 &
  pids+=($!); gpus+=("$g")
  echo "gpu$g -> ${IDS[$g]}  pid=$!"
done

# 等齐再出页。逐个记录失败的卡而不是立刻退出 —— 一张卡挂掉不该让
# 另外七条已经渲好的结果白跑，建页脚本本来就会跳过没完成的 run。
echo
echo "等 ${#pids[@]} 张卡跑完（每条约 8 分钟）…"
fail=()
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    fail+=("${gpus[$i]}")
    echo "  ✗ gpu${gpus[$i]} 失败，见 logs/infer8_text/gpu${gpus[$i]}.log" >&2
  fi
done
[ "${#fail[@]}" -eq 0 ] && echo "  ✓ 全部完成" || echo "  ${#fail[@]} 张卡失败: ${fail[*]}" >&2

[ "$NO_BUILD" = "1" ] && { echo "NO_BUILD=1，不出页"; exit 0; }

# 只喂真正产出了 generated.mp4 的 run，避免把空目录带进页面
runs=()
for g in "$@"; do
  d="output/abot_inference/${TAG}_gpu${g}"
  compgen -G "$d/*/generated.mp4" > /dev/null && runs+=("$d")
done
[ "${#runs[@]}" -gt 0 ] || { echo "没有任何完成的推理结果，不出页" >&2; exit 1; }

echo
echo "建核对页（${#runs[@]} 个样本）…"
python3 code/abot/build_text_infer_viz.py \
  --runs "${runs[@]}" --out "$OUT_HTML" --artifact-out "$ARTIFACT_HTML"

# 判据本身也要验：静态检查全绿而页面一片空白，这个坑踩过一次
python3 code/abot/check_page_js.py "$OUT_HTML"

[ "${#fail[@]}" -eq 0 ] || exit 1
