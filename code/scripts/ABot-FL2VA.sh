#!/usr/bin/env bash
# ABot-World-Explorer 500h -> MiniMax-H3 FL2VA，两阶段（先缓存 latent，再训 DiT）。
#
# 与 VRising-FL2VA.sh 的差别只有三处，其余（两阶段拆分、静音兜底、动作注入开关）
# 完全一样：
#   1. NUM_FRAMES=124 而不是 107。124 = 17*7+5 是 H3 的**原生值**（官方 example
#      清一色 124），V-Rising 被迫用 107 只是因为源 clip 只有 5.06s；这边源是
#      60s，没有这个约束，直接对齐预训练先验。
#   2. 数据是 build_abot_clips.py 预切好的 832x480 @24fps 切片，
#      所以 stage 1 里的重采样和缩放都退化成恒等（实测 0 重复帧、不丢尾帧）。
#   3. 条件走**逐 latent 动作文本注入**（ACTION_MODE=text，默认）：
#      9 位按键 -> 结构化标注 -> 文本段镜像偏移落位 + score_mod 硬绑定掩码。
#      不再建 FiLM/bias 动作层，stage 2 的 --action_num_buttons 恒为 0。
#      旧的特征空间注入用 ACTION_MODE=cond 保留，仅供对照，已实测失效。
#
# 用法:
#   bash .../ABot-FL2VA.sh 64                       # 冒烟
#   STAGE=1 bash .../ABot-FL2VA.sh 2000             # 只缓存 latent
#   NUM_PROCESSES=8 STAGE=all bash .../ABot-FL2VA.sh 7872
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/dlami/nvme/danze/minimax_finetune}"
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-$PROJECT_ROOT/DiffSynth-Studio-h3}"
cd "$DIFFSYNTH_ROOT"

# 环境里 diffsynth 是 editable 安装指向旧 checkout，且 accelerate launch 时
# sys.path[0] 是脚本目录而非仓库根，不设 PYTHONPATH 必然导入到旧库。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.cache/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_ROOT/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/.cache/triton}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.cache/tmp}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

SUBSET="${1:-64}"
STAGE="${STAGE:-all}"                # 1 | 2 | all
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
ACTION_MODE="${ACTION_MODE:-text}"    # text | cond | none
ACTION_BUTTONS="${ACTION_BUTTONS:-8}" # 仅 ACTION_MODE=cond 时有意义
ACTION_TRAIN_ONLY="${ACTION_TRAIN_ONLY:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-1}"
MAX_DATA_ITEMS="${MAX_DATA_ITEMS:-}"
# 透传额外参数给 train.py。显存紧张时用，例如：
#   EXTRA_ARGS="--fp8_models dit --use_gradient_checkpointing_offload"
# stage 2 的 ~79 GB 里大头是 transformer 权重(62 GB)，fp8 能砍掉一半。
EXTRA_ARGS="${EXTRA_ARGS:-}"

DATA_BASE="${DATA_BASE:-$PROJECT_ROOT/data/clips}"
if [ -z "${META:-}" ]; then
  case "$SUBSET" in
    7872) META="$PROJECT_ROOT/data/abot_meta_train_7872.jsonl" ;;
    # 20000 = 20128 全量减去沿用的那 128 条测试集（见 split 的 --fixed-test）
    20000) META="$PROJECT_ROOT/data/abot_meta_train_20000.jsonl" ;;
    128) META="$PROJECT_ROOT/data/abot_meta_test_128.jsonl" ;;
    *) META="$PROJECT_ROOT/data/abot_meta_${SUBSET}.jsonl" ;;
  esac
fi
OUT="${OUT:-$PROJECT_ROOT/output/minimax_h3_abot/${SUBSET}}"
NUM_FRAMES=124                        # 17*7+5 -> latent_t = 37

ACCELERATE_ARGS=()
if [ "$NUM_PROCESSES" -gt 1 ]; then
  ACCELERATE_ARGS=(--multi_gpu --num_processes "$NUM_PROCESSES" --num_machines 1)
fi

DATA_LIMIT_ARGS=()
INJECT_COMPLETENESS_ARGS=(--require-complete)
if [ -n "$MAX_DATA_ITEMS" ]; then
  DATA_LIMIT_ARGS=(--max_data_items "$MAX_DATA_ITEMS")
  INJECT_COMPLETENESS_ARGS=(--expected-count "$MAX_DATA_ITEMS")
fi

case "$ACTION_MODE" in
  text) ACTION_BUTTONS=0 ;;            # 文本注入不建动作层
  none) ACTION_BUTTONS=0 ;;
  cond)
    case "$ACTION_BUTTONS" in
      8) ACTION_INJECT_ARGS=(--active-keys-only) ;;
      11) ACTION_INJECT_ARGS=(--keys-only) ;;
      17) ACTION_INJECT_ARGS=(--cam-dropout 0) ;;
      *) echo "ACTION_MODE=cond 时 ACTION_BUTTONS 仅支持 8/11/17"; exit 1 ;;
    esac ;;
  *) echo "不支持 ACTION_MODE=$ACTION_MODE；仅支持 text/cond/none"; exit 1 ;;
esac

[ -f "$META" ] || { echo "找不到 $META，先跑 python3 /opt/dlami/nvme/danze/minimax_finetune/code/abot/build_abot_clips.py --num-clips $SUBSET"; exit 1; }
echo "== subset=$SUBSET  stage=$STAGE  $(wc -l < "$META") 条  num_frames=$NUM_FRAMES  action=$ACTION_MODE  processes=$NUM_PROCESSES =="

# ---------- stage 1: 预算 latent（text encoder + video VAE + audio VAE）----------
if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
accelerate launch "${ACCELERATE_ARGS[@]}" examples/minimax_h3/model_training/train.py \
  --dataset_base_path "$DATA_BASE" \
  --dataset_metadata_path "$META" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image" \
  --height 480 \
  --width 832 \
  --num_frames $NUM_FRAMES \
  --dataset_repeat 1 \
  --dataset_num_workers "$DATASET_NUM_WORKERS" \
  "${DATA_LIMIT_ARGS[@]}" \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUT}-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --silent_on_missing_audio \
  --enable_tensorboard_log \
  --task "sft:data_process"
echo "== stage 1 完成: ${OUT}-cache =="
case "$ACTION_MODE" in
  text)
    # 事后重写 prompt_embeds + packed。视频/音频 latent 不动，
    # 所以换标注方案只需几分钟重跑本步，不必重跑 stage 1 的 VAE 编码。
    python3 "$PROJECT_ROOT/code/abot/inject_abot_text.py" \
      --meta "$META" --cache "${OUT}-cache"
    echo "== 逐 latent 文本条件注入完成 =="
    ;;
  cond)
    python3 "$PROJECT_ROOT/code/abot/inject_abot_action.py" \
      --meta "$META" --cache "${OUT}-cache" \
      "${INJECT_COMPLETENESS_ARGS[@]}" "${ACTION_INJECT_ARGS[@]}"
    echo "== 动作张量注入完成: dim=$ACTION_BUTTONS =="
    ;;
esac
fi

# ---------- stage 2: 训 DiT LoRA ----------
if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
[ -d "${OUT}-cache" ] || { echo "找不到 stage 1 的 cache ${OUT}-cache"; exit 1; }
accelerate launch "${ACCELERATE_ARGS[@]}" examples/minimax_h3/model_training/train.py \
  --dataset_base_path "${OUT}-cache" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image" \
  --height 480 \
  --width 832 \
  --num_frames $NUM_FRAMES \
  --dataset_repeat 1 \
  --dataset_num_workers "$DATASET_NUM_WORKERS" \
  "${DATA_LIMIT_ARGS[@]}" \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs $NUM_EPOCHS \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --silent_on_missing_audio \
  --enable_tensorboard_log \
  ${SAVE_STEPS:+--save_steps $SAVE_STEPS} \
  $( [ "$ACTION_BUTTONS" -gt 0 ] && echo "--action_num_buttons $ACTION_BUTTONS" ) \
  $( [ "$ACTION_TRAIN_ONLY" = "1" ] && echo "--action_train_only" ) \
  $EXTRA_ARGS \
  --task "sft:train"
echo "== stage 2 完成: $OUT =="
fi
