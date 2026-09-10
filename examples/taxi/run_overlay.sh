#!/usr/bin/env bash
# Convert a taxi action file to an action matrix, run inference, then burn a
# WASDIJKL key overlay onto the result -- the three-step pipeline described
# in examples/taxi/convert_actions.py and examples/overlay_keys.py.
#
# Run from the H3-World repo root:
#   examples/taxi/run_overlay.sh [straight.json|left.json|right.json]
set -euo pipefail

ACTIONS_FILE="${1:-straight.json}"
STEM="${ACTIONS_FILE%.json}"

TAXI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TAXI_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Matches README.md's Inference section; override via env if your checkpoint
# or first-frame image live elsewhere.
CHECKPOINT="${H3WORLD_CHECKPOINT:-checkpoints/H3-World/step-10000.safetensors}"
FIRST_FRAME="${H3WORLD_FIRST_FRAME:-examples/taxi/taxi.png}"
SCENE_PROMPT="$(cat examples/taxi/prompt.txt)"

if [[ "$STEM" == "0001" ]]; then
  ACTIONS_NPY="examples/taxi/actions.npy"
else
  ACTIONS_NPY="examples/taxi/${STEM}_actions.npy"
fi

echo "== Converting $ACTIONS_FILE to action matrix =="
CONVERT_OUT="$(python3 examples/taxi/convert_actions.py --actions-file "$ACTIONS_FILE")"
echo "$CONVERT_OUT"
NUM_FRAMES="$(echo "$CONVERT_OUT" | grep -- '--num-frames' | awk '{print $2}')"

VIDEO_OUT="outputs/taxi_${STEM}.mp4"
OVERLAY_OUT="outputs/taxi_${STEM}_overlay.mp4"

echo "== Running inference ($NUM_FRAMES frames) =="
python3 code/abot/infer.py \
  --checkpoint "$CHECKPOINT" \
  --first-frame "$FIRST_FRAME" \
  --scene-prompt "$SCENE_PROMPT" \
  --action-file "$ACTIONS_NPY" \
  --num-frames "$NUM_FRAMES" \
  --subject car \
  --out "$VIDEO_OUT"

echo "== Burning key overlay =="
python3 examples/overlay_keys.py --video "$VIDEO_OUT" --actions "$ACTIONS_NPY" --out "$OVERLAY_OUT"

echo "Done: $OVERLAY_OUT"
