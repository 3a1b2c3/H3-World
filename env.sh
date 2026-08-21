# 本目录统一的环境入口。所有缓存都指向 nvme —— 根盘 / 只有 100G，
# torch/triton 的编译缓存默认落在 /tmp 和 ~/.triton，都在根盘上。
export DIFFSYNTH_ROOT=/opt/dlami/nvme/danze/minimax_finetune/DiffSynth-Studio-h3
export PATH=/opt/dlami/nvme/danze/envs/minimax_h3/bin:$PATH
export HF_HOME=/opt/dlami/nvme/danze/minimax_finetune/.cache/hf
export TORCHINDUCTOR_CACHE_DIR=/opt/dlami/nvme/danze/minimax_finetune/.cache/torchinductor
export TRITON_CACHE_DIR=/opt/dlami/nvme/danze/minimax_finetune/.cache/triton
export XDG_CACHE_HOME=/opt/dlami/nvme/danze/minimax_finetune/.cache/xdg
