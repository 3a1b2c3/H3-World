#!/bin/bash
# Install H3-World's environment and run the README's example inference
# command. Model weights are downloaded by the separate download_models.sh
# (called below unless --skip-download). Deviates from the README in one
# way: uses a plain `python3 -m venv` + pip instead of conda -- whatever
# python3 is already on PATH, not pinned to 3.10 like the README's conda
# instructions (README notes it was "tested with Python 3.10", not that
# other versions are known broken).
#
#   bash setup_and_run.sh              # install + download + run the example
#   bash setup_and_run.sh --skip-install   # already installed, just download+run
#   bash setup_and_run.sh --skip-download  # already downloaded, just run
#   HF_TOKEN=hf_xxx bash setup_and_run.sh  # if either HF repo needs auth
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SKIP_INSTALL=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

echo "=== H3-World setup: $(date) ==="
echo

if [ "$SKIP_INSTALL" = "0" ]; then
  echo "--- Installing environment ---"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is not on PATH." >&2
    exit 1
  fi

  if [ ! -d "$REPO_ROOT/.venv" ]; then
    python3 -m venv "$REPO_ROOT/.venv"
  fi
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
  pip install --upgrade pip

  echo "Installing torch (cu132 -- README says cu128, using cu132 for this box's CUDA 13.2)..."
  pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu132

  echo "Installing requirements.txt..."
  pip install -r requirements.txt

  if [ ! -d "$REPO_ROOT/DiffSynth-Studio-h3-v2" ]; then
    echo "Cloning DiffSynth-Studio at the pinned commit..."
    git clone https://github.com/modelscope/DiffSynth-Studio.git DiffSynth-Studio-h3-v2
    git -C DiffSynth-Studio-h3-v2 checkout "$(cat code/diffsynth_base_commit.txt)"
    echo "Applying H3-World's action-attention patch..."
    git -C DiffSynth-Studio-h3-v2 apply ../code/diffsynth_h3_action.patch
  else
    echo "DiffSynth-Studio-h3-v2 already present, skipping clone."
    echo "  (if this is a partial/broken checkout, remove it and re-run without --skip-install)"
  fi
  # Explicitly NOT pip-installed in editable/-e mode -- the H3-World scripts
  # import it via ABOT_DIFFSYNTH_ROOT/DIFFSYNTH_ROOT (env.sh) as a plain
  # checked-out directory on sys.path, not an installed package. Installing
  # it separately would risk a second, unpatched copy shadowing this one.
  echo
else
  echo "--- Skipping install (--skip-install) ---"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
  echo
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/env.sh"

if [ "$SKIP_DOWNLOAD" = "0" ]; then
  bash "$REPO_ROOT/download_models.sh"
  echo
else
  echo "--- Skipping download (--skip-download) ---"
  echo
fi

echo "--- Running example inference ---"
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

echo
echo "Done. Output: outputs/example_forward.mp4"
