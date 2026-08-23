#!/usr/bin/env bash
# 逐 latent 动作文本注入 —— 一条命令跑完：探针 -> 数据构建 -> 校验 -> 起训。
#
# 每一步不过就停，绝不带着坏数据往下走。这套方案出错不会当场报警，
# 只会在训练里静默表现为「学不到绑定」，所以宁可多卡几道。
#
# 用法:
#   bash code/scripts/run_text_pipeline.sh                # 全流程
#   STOP_AFTER=verify bash code/scripts/run_text_pipeline.sh   # 只到校验，不起训
#   SKIP_DATA=1 bash code/scripts/run_text_pipeline.sh    # 数据已就绪，只起训
#   NUM_EPOCHS=3 bash code/scripts/run_text_pipeline.sh   # 改 epoch 数
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/dlami/nvme/danze/minimax_finetune}"
CONDA_ENV="${CONDA_ENV:-minimax_h3}"
META="${META:-$PROJECT_ROOT/data/abot_meta_train_7872.jsonl}"
CACHE="${CACHE:-$PROJECT_ROOT/output/minimax_h3_abot/7872-cache}"
OUT="${OUT:-$PROJECT_ROOT/output/minimax_h3_abot/7872_text}"
NUM_GPUS="${NUM_GPUS:-8}"
SUBSET="${SUBSET:-7872}"              # ABot-FL2VA.sh 用它挑 metadata，必须与 META 对应
NUM_EPOCHS="${NUM_EPOCHS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
STOP_AFTER="${STOP_AFTER:-}"          # probe | data | verify | 空=全跑
SKIP_DATA="${SKIP_DATA:-0}"

cd "$PROJECT_ROOT"
mkdir -p logs/pipeline
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$1" >&2; exit 1; }

# ── 0. 前置检查 ────────────────────────────────────────────────────────────
step "0/4  前置检查"
[ -f "$META" ]   || die "找不到 metadata: $META"
[ -d "$CACHE" ]  || die "找不到 stage 1 缓存: $CACHE（先跑 STAGE=1 的 ABot-FL2VA.sh）"
N_PTH=$(find "$CACHE" -name '*.pth' | wc -l)
N_META=$(wc -l < "$META")
echo "  metadata $N_META 行，缓存 $N_PTH 条"
[ "$N_PTH" -eq "$N_META" ] || die "缓存条数与 metadata 不符"
FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{t+=$1} END{print t+0}')
echo "  GPU 已占用合计 ${FREE} MiB"
[ "$FREE" -lt 5000 ] || die "GPU 上还有别的进程占着显存，先清干净"

# ── 1. 掩码落点探针 ────────────────────────────────────────────────────────
# 掩码是这套方案的承重墙，而且它错了不报错。改过 DiT 就必须先过这一关。
step "1/4  掩码落点探针（6 条判据）"
set +e
CUDA_VISIBLE_DEVICES=0 python3 code/abot/probe_action_mask.py > logs/pipeline/probe.log 2>&1
PROBE_RC=$?
set -e
grep -E '^\s+\[|全部通过|失败判据' logs/pipeline/probe.log | sed 's/^/  /' || true
[ "$PROBE_RC" -eq 0 ] || die "掩码探针未通过，不要往下走（见 logs/pipeline/probe.log）"
[ "$STOP_AFTER" = "probe" ] && { echo "STOP_AFTER=probe，到此为止"; exit 0; }

# ── 2. 数据构建 ────────────────────────────────────────────────────────────
if [ "$SKIP_DATA" = "1" ]; then
  step "2/4  数据构建（SKIP_DATA=1，跳过）"
else
  step "2/4  数据构建（$NUM_GPUS 卡分片，约 10 分钟）"
  echo "  重写 prompt_embeds + packed；视频/音频 latent 不动"
  pids=()
  for ((g = 0; g < NUM_GPUS; g++)); do
    python3 code/abot/inject_abot_text.py \
      --meta "$META" --cache "$CACHE" \
      --device "cuda:$g" --shard "$g" --num-shards "$NUM_GPUS" \
      > "logs/pipeline/inject_shard$g.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "  ✗ shard $i 失败，见 logs/pipeline/inject_shard$i.log"; fail=1; }
  done
  [ "$fail" -eq 0 ] || die "数据构建有分片失败"
  grep -h "统一 padding" logs/pipeline/inject_shard0.log | sed 's/^/  /'
  grep -h "real_used\|seq_len " logs/pipeline/inject_shard0.log | tail -2 | sed 's/^/  /'
  echo "  ✓ $NUM_GPUS 个分片全部完成"
fi
[ "$STOP_AFTER" = "data" ] && { echo "STOP_AFTER=data，到此为止"; exit 0; }

# ── 3. 全量校验 ────────────────────────────────────────────────────────────
step "3/4  缓存完整性校验（8 条判据，逐条核不抽查）"
rm -rf .cache/verify && mkdir -p .cache/verify
for ((i = 0; i < NUM_GPUS; i++)); do
  python3 code/abot/verify_text_cache.py --cache "$CACHE" \
    --shard "$i" --num-shards "$NUM_GPUS" --out ".cache/verify/$i.json" \
    > "logs/pipeline/verify$i.log" 2>&1 &
done
wait
python3 - <<'PY' || die "校验未通过"
import json, glob, sys
S = [json.load(open(f)) for f in sorted(glob.glob('.cache/verify/*.json'))]
n = sum(s['n'] for s in S); ok = sum(s['ok'] for s in S)
print(f"  {ok}/{n} 通过")
for s in S:
    for name, bad in s['failures']:
        print(f"  ✗ {name}: {'; '.join(bad)}")
seq = sorted({x for s in S for x in s['seq_lens']})
off = sorted({x for s in S for x in s['offsets']})
print(f"  seq_len 取值集合  {seq}   <- 必须唯一，否则 flex_attention 会反复重编译")
print(f"  镜像偏移取值集合  {off}   <- 必须是单一负常数")
sys.exit(0 if (ok == n and len(seq) == 1 and len(off) == 1 and off[0] < 0) else 1)
PY
echo "  ✓ 全量通过"
[ "$STOP_AFTER" = "verify" ] && { echo "STOP_AFTER=verify，到此为止"; exit 0; }

# ── 4. 起训 ────────────────────────────────────────────────────────────────
step "4/4  stage 2 训练（$NUM_EPOCHS epochs）"
ln -sfn "$(basename "$CACHE")" "$(dirname "$OUT")/$(basename "$OUT")-cache"
LOG="logs/abot_$(basename "$OUT")_${NUM_GPUS}gpu.log"
echo "  输出 $OUT"
echo "  日志 $LOG"
NUM_PROCESSES="$NUM_GPUS" STAGE=2 NUM_EPOCHS="$NUM_EPOCHS" SAVE_STEPS="$SAVE_STEPS" \
OUT="$OUT" nohup bash code/scripts/ABot-FL2VA.sh "$SUBSET" > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "  pid=$TRAIN_PID，等 3 分钟确认能跑起来 …"
sleep 180
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  tail -c 3000 "$LOG" | tr '\r' '\n' | grep -vE '^\s*$' | tail -20
  die "训练进程已退出，见 $LOG"
fi
tail -c 3000 "$LOG" | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s/it\]' | tail -1 | sed 's/^/  进度 /'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ' | sed 's/^/  显存 /'
echo
echo "  ✓ 训练已在后台运行。跟进度："
echo "      tail -f $LOG | tr '\\r' '\\n' | grep -oE '[0-9]+/[0-9]+ \\[[^]]*\\]'"
