#!/usr/bin/env bash
# MiniMax-H3 V-Rising 单卡 LoRA 训练 driver（子集可配）。
#
# 实测速率：
#   stage 1  6.0 s/clip（干净卡）/ 10.6 s/clip（与人抢同一张卡）
#   stage 2  9.06 s/step
#
# 默认子集是 form_wolf(1374) 而不是 all(20699)：63M 参数的 rank-32 LoRA 在
# 1000-10000 步收敛，all x 5 epoch = 103495 步严重过量。窄切片 6870 步正好，
# 且动作空间只有 WASD、无形态突变，适合当 Stage B 动作注入的验证床。
#
# 之所以要这个 driver 而不是直接跑 .sh：
#   1. 长跑中途被 OOM / 抢卡 / 网络抖动打断的概率不低。runner.py 已打了断点续跑
#      补丁（.tmp + 原子 rename），直接重启就能接上，这里做最多 MAX_RETRY 次自动重启。
#      换 GPU 也靠这条：kill 掉带新的 GPU= 重启即可，已缓存的不会重算。
#   2. stage 2 设了 SAVE_STEPS，不用等满一个 epoch 才有权重。
#   3. 两阶段串联，stage 1 成功才进 stage 2。
#
# 用法:  GPU=1 SUBSET=form_wolf setsid nohup bash /nfs/danze/run_h3_all.sh \
#            > /nfs/danze/logs/h3_form_wolf_nohup.log 2>&1 &
# 中断:  kill $(cat /nfs/danze/logs/h3_form_wolf.pid)
# 可选子集: form_wolf(1374) wolf_life(5430) transform(9314) all(20699) smoke(64)
set -uo pipefail

REPO=/nfs/danze/repo/DiffSynth-Studio-new
LOGDIR=/nfs/danze/logs
GPU=${GPU:-1}
SUBSET=${SUBSET:-form_wolf}
MAX_RETRY=${MAX_RETRY:-5}
# SAVE_STEPS 要和总步数配套：form_wolf 是 1374x5=6870 步，2000 步一存只有 3 个
# checkpoint，看不出早停时机，所以窄切片用 500。all(103495 步) 才该用 2000。
export SAVE_STEPS=${SAVE_STEPS:-500}

DATA_BASE=/nfs/danze/data/v_rising/vrising_train_bundle/data/transform_classified
META="$DATA_BASE/h3_meta_${SUBSET}.jsonl"
[ -f "$META" ] || { echo "找不到 $META，先跑 python3 /nfs/danze/data/v_rising/build_h3_metadata.py"; exit 1; }
N=$(wc -l < "$META")                      # 子集条数，stage 1 完成判据

mkdir -p "$LOGDIR"
LOG=$LOGDIR/h3_${SUBSET}.log
echo $$ > "$LOGDIR/h3_${SUBSET}.pid"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU

say "=========== 训练启动 (subset=$SUBSET, $N 条, GPU$GPU, SAVE_STEPS=$SAVE_STEPS) ==========="
# 按实测 6.0 s/clip（干净卡）/ 10.6 s/clip（争抢）与 9.06 s/step 外推
say "预计 stage1 ~$((N * 6 / 3600))-$((N * 11 / 3600))h + stage2(5ep) ~$((N * 5 * 9 / 3600))-$((N * 5 * 18 / 3600))h。断点续跑已启用。"

# ---------- 占卡脚本让路 ----------
for p in $(pgrep -f "finetune_minimax_h3\.py" 2>/dev/null); do
  c=$(cat "/proc/$p/comm" 2>/dev/null) || continue
  case "$c" in python*) ;; *) continue ;; esac
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "-g $GPU" || continue
  say "SIGTERM 占卡脚本 PID=$p"
  kill -TERM "$p"
  for _ in $(seq 60); do kill -0 "$p" 2>/dev/null || break; sleep 2; done
done
say "GPU$GPU 可用显存: $(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits) MiB"

# ---------- stage 1（可重试）----------
CACHE=/nfs/danze/model/minimax_h3_vrising/${SUBSET}-cache
for try in $(seq 1 "$MAX_RETRY"); do
  n=$(find "$CACHE" -name '*.pth' 2>/dev/null | wc -l)
  say "stage 1 第 $try 次尝试（已缓存 $n / $N）"
  STAGE=1 bash examples/minimax_h3/model_training/lora/VRising-FL2VA.sh "$SUBSET" \
    >> "$LOGDIR/h3_${SUBSET}_stage1.log" 2>&1
  rc=$?
  n=$(find "$CACHE" -name '*.pth' 2>/dev/null | wc -l)
  if [ $rc -eq 0 ] && [ "$n" -ge "$N" ]; then
    say "✅ stage 1 完成：$n 条，$(du -sh "$CACHE" 2>/dev/null | cut -f1)"
    break
  fi
  say "⚠️ stage 1 中断 rc=$rc，已缓存 $n 条；$( [ $try -lt $MAX_RETRY ] && echo '60s 后重试（会跳过已完成的）' || echo '重试用尽' )"
  tail -15 "$LOGDIR/h3_${SUBSET}_stage1.log" | tr '\r' '\n' | tail -8 | tee -a "$LOG"
  [ $try -lt "$MAX_RETRY" ] && sleep 60
done

n=$(find "$CACHE" -name '*.pth' 2>/dev/null | wc -l)
if [ "$n" -lt "$N" ]; then
  say "❌ stage 1 未能完成（$n / $N），不进 stage 2。"
  exit 1
fi

# ---------- stage 2 ----------
say "启动 stage 2（5 epoch，每 $SAVE_STEPS 步存一次）"
STAGE=2 bash examples/minimax_h3/model_training/lora/VRising-FL2VA.sh "$SUBSET" \
  >> "$LOGDIR/h3_${SUBSET}_stage2.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  say "✅ stage 2 完成，权重: /nfs/danze/model/minimax_h3_vrising/$SUBSET"
  ls -la "/nfs/danze/model/minimax_h3_vrising/$SUBSET" 2>/dev/null | tee -a "$LOG"
else
  say "❌ stage 2 失败 rc=$rc，尾部日志："
  tail -30 "$LOGDIR/h3_${SUBSET}_stage2.log" | tr '\r' '\n' | tail -20 | tee -a "$LOG"
fi
say "=========== driver 结束 ==========="
