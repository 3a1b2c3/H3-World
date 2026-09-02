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

echo "Installing torch (cu132, unpinned)..."
# CONFIRMED on real hardware: torch==2.10.0 (the README's pinned version)
# does NOT exist on the cu132 index at all (versions jump 2.0.x -> 2.12.x+)
# -- pinning to it silently fell back to a CPU-only build, not an error.
# Unpinned here so pip picks whatever torch version cu132 actually has.
# --force-reinstall: a stray CPU-only torch from an earlier troubleshooting
# attempt can read as "already satisfied" and never get replaced without
# this (pip's version matching doesn't distinguish +cu132 vs +cpu local
# build tags). Symptom if this ever regresses: "Torch not compiled with
# CUDA enabled" at generation time, not at install time.
pip install torch --index-url https://download.pytorch.org/whl/cu132 --force-reinstall

echo "Installing torchvision + torchaudio (CPU-only)..."
# Neither is in the README's install steps at all, but
# DiffSynth-Studio-h3-v2 imports torchaudio unconditionally (see
# diffsynth/utils/data/audio.py) for basic format utilities (not GPU
# compute), and torchvision is used similarly here -- CPU-only sidesteps
# needing an exact-matching CUDA build for either, which the cu132 index
# doesn't reliably have (see the torch note above). --no-deps on both:
# without it, pip would each try to pull in their own CPU-only torch
# dependency and clobber the working CUDA torch installed above.
# --force-reinstall: same stale-reinstall reasoning as torch above --
# .venv is reused across runs, so a leftover mismatched-GPU-build
# torchvision/torchaudio from an earlier attempt won't get replaced
# without it.
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

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
