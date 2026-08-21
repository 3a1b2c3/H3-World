#!/usr/bin/env bash
# V-Rising 形态转换 LoRA — MiniMax-H3 FL2VA，两阶段（先缓存 latent，再训 DiT）。
#
# 为什么用 FL2VA：transform 类 clip 的首帧是形态 A、尾帧是形态 B，
# input_image/end_image 正好由 train.py 自动从训练视频里取首尾帧，
# 模型学的就是「A 形态 -> B 形态」的变身过程，和数据天然对齐。
#
# 用法:
#   bash examples/minimax_h3/model_training/lora/VRising-FL2VA.sh smoke   # 64 条冒烟
#   bash examples/minimax_h3/model_training/lora/VRising-FL2VA.sh         # 9314 条 transform
#   bash examples/minimax_h3/model_training/lora/VRising-FL2VA.sh all     # 20699 条全量
#
# 可以只跑其中一个阶段（默认两个都跑）：
#   STAGE=1 bash ... VRising-FL2VA.sh smoke      # 只缓存 latent，不需要 transformer 权重
#   STAGE=2 bash ... VRising-FL2VA.sh smoke      # 只训 DiT，要求 stage 1 的 cache 已存在
# 分开跑的意义：transformer 有 66 GB 要下，stage 1 只用编码器（78 GB）。
# 先跑 STAGE=1 验证通路，同时后台下 transformer，不用干等。
set -euo pipefail
cd "$(dirname "$0")/../../../.."     # -> DiffSynth-Studio-new 根目录

# 必须显式置顶本仓库：环境里 diffsynth 是 editable 安装，指向旧 checkout
# /data/danzechen/DiffSynth-Studio，而 accelerate launch 时 sys.path[0] 是脚本
# 所在目录而不是这里，不设 PYTHONPATH 会导入到旧库，
# 直接 ModuleNotFoundError: diffsynth.utils.data.minimax_h3。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

SUBSET="${1:-transform}"
STAGE="${STAGE:-all}"                # 1 | 2 | all
# SAVE_STEPS：每 N 步存一次 LoRA，不设则按 epoch 存。
# 全量 20699 条时单个 epoch 要 ~52h，只按 epoch 存意味着两天才有第一个可用权重，
# 出了问题也没法early stop，所以长跑务必设上。
SAVE_STEPS="${SAVE_STEPS:-}"
# 动作条件（Stage B）。ACTION_BUTTONS=12 打开注入，前提是 stage 1 的 cache 已经用
# /nfs/danze/inject_action_into_cache.py 注入过 action_cond。
# ACTION_TRAIN_ONLY=1 只训 action_embedders、冻结其余，用来验证注入通路：
# 配合零初始化，loss 曲线应与纯 LoRA baseline 起点重合。
ACTION_BUTTONS="${ACTION_BUTTONS:-0}"
ACTION_TRAIN_ONLY="${ACTION_TRAIN_ONLY:-0}"
DATA_BASE="/nfs/danze/data/v_rising/vrising_train_bundle/data/transform_classified"
META="$DATA_BASE/h3_meta_${SUBSET}.jsonl"
OUT="/nfs/danze/model/minimax_h3_vrising/${SUBSET}"

[ -f "$META" ] || { echo "找不到 $META，先跑 python3 /nfs/danze/data/v_rising/build_h3_metadata.py"; exit 1; }

# 107 = 17*6+5，是 17n+5 里不超过可用帧数的最大值。
# 原 clip 81帧@16fps=5.06s，重采样到 24fps 只有 121 帧可用，124 会被自动降级，
# 不如直接写死 107 保证 batch 内帧数一致。
NUM_FRAMES=107

echo "== subset=$SUBSET  stage=$STAGE  meta=$META  $(wc -l < "$META") 条 =="

# ---------- stage 1: 预算 latent（text encoder + video VAE + audio VAE）----------
# 只跑一遍，把结果缓存到磁盘；20699 条约需上百 GB，注意留空间。
if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path "$DATA_BASE" \
  --dataset_metadata_path "$META" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames $NUM_FRAMES \
  --dataset_repeat 1 \
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
echo "== stage 1 完成，latent cache: ${OUT}-cache =="
fi

# ---------- stage 2: 训 DiT LoRA ----------
if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
[ -d "${OUT}-cache" ] || { echo "找不到 stage 1 的 cache ${OUT}-cache，先跑 STAGE=1"; exit 1; }
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path "${OUT}-cache" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames $NUM_FRAMES \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --silent_on_missing_audio \
  --enable_tensorboard_log \
  ${SAVE_STEPS:+--save_steps $SAVE_STEPS} \
  $( [ "$ACTION_BUTTONS" -gt 0 ] && echo "--action_num_buttons $ACTION_BUTTONS" ) \
  $( [ "$ACTION_TRAIN_ONLY" = "1" ] && echo "--action_train_only" ) \
  --task "sft:train"
echo "== stage 2 完成，LoRA 权重: $OUT =="
fi
