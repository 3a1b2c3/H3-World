# Shared environment entry point for this repo. Source it before running
# any script here: `source env.sh`.
#
# All caches point inside the repo rather than the default locations
# (usually /tmp and ~/.triton) so a small root filesystem doesn't fill up
# and everything lives next to the code that produced it.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export DIFFSYNTH_ROOT="$REPO_ROOT/DiffSynth-Studio-h3-v2"
export ABOT_DIFFSYNTH_ROOT="$DIFFSYNTH_ROOT"
export HF_HOME="$REPO_ROOT/.cache/hf"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache/triton"
export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"
