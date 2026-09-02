#!/bin/bash
# Download H3-World's model weights: the MiniMax-H3 backbone (~135 GiB) and
# the H3-World LoRA checkpoint. Requires env.sh to already be sourced (sets
# DIFFSYNTH_ROOT/HF_HOME) and the venv from setup_and_run.sh to be active.
#
#   source env.sh && source .venv/bin/activate && bash download_models.sh
#   HF_TOKEN=hf_xxx bash download_models.sh   # if either HF repo needs auth
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ -z "${DIFFSYNTH_ROOT:-}" ]; then
  echo "ERROR: DIFFSYNTH_ROOT is not set -- run 'source env.sh' first." >&2
  exit 1
fi

# Backbone download is ~135 GiB; LoRA checkpoint is small (65.6M params,
# well under 1 GB at fp16). Warn loudly rather than fail 2 hours into a
# download.
AVAIL_GB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print int($4/1024/1024)}')
echo "Available disk at $REPO_ROOT: ${AVAIL_GB} GiB"
if [ "$AVAIL_GB" -lt 160 ]; then
  echo "WARNING: backbone download is ~135 GiB; ${AVAIL_GB} GiB free may not be enough" \
       "once caches are included. Continuing anyway -- Ctrl+C now if you want to free" \
       "space first."
fi
echo

echo "Backbone: MiniMax/MiniMax-H3 (~135 GiB) -- this will take a while."
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("MiniMax/MiniMax-H3", local_dir="${DIFFSYNTH_ROOT}/models/MiniMax/MiniMax-H3")
PY

echo
echo "LoRA checkpoint: DANNY621/H3-World/step-10000.safetensors..."
python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("DANNY621/H3-World", "step-10000.safetensors", local_dir="checkpoints/H3-World")
PY

echo
echo "Done. Backbone: ${DIFFSYNTH_ROOT}/models/MiniMax/MiniMax-H3"
echo "      LoRA:     checkpoints/H3-World/step-10000.safetensors"
