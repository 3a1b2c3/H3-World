#!/bin/bash
# Run H3-World's example inferences: the README's forward-preset example,
# plus the racer example (a real per-frame action sequence, not a preset).
# Assumes setup.sh has already been run (venv + DiffSynth-Studio checkout
# in place). Downloads model weights first via download_models.sh unless
# --skip-download.
#
#   bash run.sh                # download (if needed) + run both examples
#   bash run.sh --skip-download    # already downloaded, just run
#   HF_TOKEN=hf_xxx bash run.sh    # if either HF repo needs auth
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [ ! -d "$REPO_ROOT/.venv" ]; then
  echo "ERROR: .venv not found -- run 'bash setup.sh' first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
# shellcheck disable=SC1091
source "$REPO_ROOT/env.sh"

if [ "$SKIP_DOWNLOAD" = "0" ]; then
  bash "$REPO_ROOT/download_models.sh"
  echo
else
  echo "--- Skipping download (--skip-download) ---"
  echo
fi

echo "--- Running example inference (README's forward-preset example) ---"
mkdir -p outputs
python3 code/abot/infer.py \
  --checkpoint checkpoints/H3-World/step-10000.safetensors \
  --first-frame examples/first_frame.png \
  --scene-prompt "A man in a yellow floral shirt stands in a dim, multi-level concrete parking garage." \
  --action-preset forward \
  --seed 2 \
  --steps 50 \
  --num-frames 124 \
  --cfg-scale 1.0 \
  --out outputs/example_forward.mp4
echo "Done. Output: outputs/example_forward.mp4"

echo "Building key overlay for the forward-preset example..."
python3 -c "
import sys
sys.path.insert(0, 'examples')
import numpy as np
from overlay_keys import build_keys9_from_preset
np.save('outputs/example_forward_actions.npy', build_keys9_from_preset(('W',), 124))
"
python3 examples/overlay_keys.py \
  --video outputs/example_forward.mp4 \
  --actions outputs/example_forward_actions.npy \
  --out outputs/example_forward_overlay.mp4
echo "Done. Output: outputs/example_forward_overlay.mp4"
echo

echo "--- Running racer example (real per-frame action sequence, not a preset) ---"
echo "Converting examples/racer/0001.json -> examples/racer/actions.npy..."
CONVERT_OUT=$(python3 examples/racer/convert_actions.py)
echo "$CONVERT_OUT"
RACER_NUM_FRAMES=$(echo "$CONVERT_OUT" | grep -- '--num-frames' | awk '{print $2}')
if [ -z "$RACER_NUM_FRAMES" ]; then
  echo "ERROR: could not parse --num-frames out of convert_actions.py's output" >&2
  exit 1
fi
python3 code/abot/infer.py \
  --checkpoint checkpoints/H3-World/step-10000.safetensors \
  --first-frame examples/racer/Screenshot.png \
  --scene-prompt "$(cat examples/racer/prompt.txt)" \
  --action-file examples/racer/actions.npy \
  --num-frames "$RACER_NUM_FRAMES" \
  --seed 2 \
  --steps 50 \
  --cfg-scale 1.0 \
  --out outputs/example_racer.mp4
echo "Done. Output: outputs/example_racer.mp4"

echo "Building key overlay for the racer example..."
python3 examples/overlay_keys.py \
  --video outputs/example_racer.mp4 \
  --actions examples/racer/actions.npy \
  --out outputs/example_racer_overlay.mp4
echo "Done. Output: outputs/example_racer_overlay.mp4"

echo
echo "Done. Outputs: outputs/example_forward.mp4, outputs/example_forward_overlay.mp4,"
echo "               outputs/example_racer.mp4, outputs/example_racer_overlay.mp4"
