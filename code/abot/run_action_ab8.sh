#!/usr/bin/env bash
# 同首帧、换 action 的对照实验：8 卡各跑一个按键预设，其余变量全部固定。
#
# 这是**粗粒度**判据 —— 只回答"换 action 出不出不同且符合的结果"。
# 回答不了"第 k 条标注绑不绑第 k 帧"，那要逐 latent 只改一条，是另一个工具。
#
# 输出走独立 TAG，不碰任何训练数据和已有推理结果。
#
# 用法:
#   bash code/abot/run_action_ab8.sh                    # 默认样本，8 卡 8 预设
#   SID=<sample_id> bash code/abot/run_action_ab8.sh    # 换一条首帧
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=${CKPT:-output/minimax_h3_abot/7872_text/step-9840.safetensors}
SID=${SID:-a3ad9c24bda131dfa0ea18efe44a4e8b}
TAG=${TAG:-ab8_step9840}
CFG=${CFG:-1.0}
PRESETS=(still forward back strafe-left strafe-right pan-left pan-right pan-right-fast)

mkdir -p logs/action_ab8
pids=(); names=()
for g in 0 1 2 3 4 5 6 7; do
  p="${PRESETS[$g]}"
  python3 code/abot/infer_abot.py \
    --checkpoint "$CKPT" \
    --sample-id "$SID" \
    --device "cuda:$g" \
    --cfg-scale "$CFG" \
    --action-preset "$p" \
    --run-name "${TAG}_${p}" \
    --overwrite \
    > "logs/action_ab8/${p}.log" 2>&1 &
  pids+=($!); names+=("$p")
  echo "gpu$g -> $p"
done

echo
echo "同首帧 $SID，8 个按键预设，等约 8 分钟 …"
fail=()
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { fail+=("${names[$i]}"); echo "  ✗ ${names[$i]} 失败" >&2; }
done
[ "${#fail[@]}" -eq 0 ] && echo "  ✓ 8 个预设全部完成" || echo "  ${#fail[@]} 个失败: ${fail[*]}" >&2
echo
echo "产物:"
for p in "${PRESETS[@]}"; do
  f=$(ls "output/abot_inference/${TAG}_${p}"/*/generated.mp4 2>/dev/null | head -1)
  [ -n "$f" ] && printf "  %-16s %s\n" "$p" "$f"
done
[ "${#fail[@]}" -eq 0 ] || exit 1
