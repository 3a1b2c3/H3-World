#!/bin/bash
# Install H3-World's Python environment (venv + torch/torchvision/torchaudio
# + requirements.txt) and clone/patch DiffSynth-Studio at the pinned
# commit. Does NOT download model weights (see download_models.sh) or run
# anything (see run.sh). Deviates from the README in one way: uses a plain
# `python3 -m venv` + pip instead of conda -- whatever python3 is already
# on PATH, not pinned to 3.10 like the README's conda instructions (README
# notes it was "tested with Python 3.10", not that other versions are known
# broken).
#
#   bash setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== H3-World setup: $(date) ==="
echo

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

echo "Installing torch (cu132 stable, pinned per README) + torchvision..."
# --force-reinstall: CONFIRMED on real hardware -- plain `pip install
# torch==2.10.0` can read as "already satisfied" against a stray CPU-only
# torch left over from an earlier troubleshooting attempt (pip's `==2.10.0`
# constraint doesn't distinguish local build tags like +cu132 vs +cpu), and
# silently keep the CPU build instead of replacing it. Symptom: "Torch not
# compiled with CUDA enabled" at generation time, not at install time.
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu132 --force-reinstall

echo "Installing torchaudio (CPU-only)..."
# torchaudio isn't in the README's install steps at all, but
# DiffSynth-Studio-h3-v2's diffsynth/utils/data/audio.py imports it
# unconditionally (same transitive-import situation as pandas below) --
# only for basic format utilities (convert_to_stereo/resample_waveform),
# not GPU compute, so CPU-only is fine here. CONFIRMED on real hardware:
# the stable cu132 index has no torchaudio build matching
# torch==2.10.0+cu132's ABI -- pip silently resolved a mismatched GPU
# build, which failed to load its compiled extension (_torchaudio.abi3.so
# couldn't find libc10_cuda.so). The CPU wheel sidesteps that entirely
# (no libc10_cuda.so dependency at all) instead of needing a matched
# torch/torchvision/torchaudio nightly triplet.
# --force-reinstall: CONFIRMED on real hardware that plain `pip install`
# here is not enough -- .venv is reused across runs (not recreated if it
# already exists), so a stale mismatched-GPU-build torchaudio from an
# earlier attempt reads as "already satisfied" and never gets replaced
# without this. --no-deps avoids pip also pulling in a CPU-only torch and
# clobbering the working CUDA torch install above.
pip install torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

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
  echo "  (if this is a partial/broken checkout, remove it and re-run setup.sh)"
fi
# Explicitly NOT pip-installed in editable/-e mode -- the H3-World scripts
# import it via ABOT_DIFFSYNTH_ROOT/DIFFSYNTH_ROOT (env.sh) as a plain
# checked-out directory on sys.path, not an installed package. Installing
# it separately would risk a second, unpatched copy shadowing this one.

echo
echo "Done. Activate with: source .venv/bin/activate"
echo "Next: bash download_models.sh   (or run.sh, which calls it automatically)"
