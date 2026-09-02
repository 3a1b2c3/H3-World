#!/usr/bin/env python3
"""Convert 0001.json's discrete per-tick {"move", "view"} actions into the
raw [num_frames, 17] action matrix `abot_action.py`'s bin_to_latent() /
action_script.py's keys9() expect -- the SAME format/pipeline used for real
training data, so this racer clip is scripted through identical code to
everything else, not a one-off reimplementation.

0001.json's moves/views (verified exhaustively: {"go forward", "no-op"} /
{"turn left", "turn right", "no-op"}) map onto three of ACTION_DIM=17's 11
binary key columns:
    "go forward" -> W
    "turn left"  -> J   (action_script.py's PAN_KEY: J = left)
    "turn right" -> L   (PAN_KEY: L = right)
Every other key (A, S, D, Q, E, I, K, Space) and all 6 continuous
rotation/translation columns are left at 0 -- 0001.json has no ground-truth
magnitude data (no COLMAP reconstruction for this clip, see
build_pose_npz.py's docstring), so there's nothing real to put there. This
means the "F" (fast-pan) 9th bit derived downstream in keys9() will never
fire (no yaw-rate signal to threshold against) -- turns will always read as
"pans left/right slowly", never "sharply". That's a real, known limitation
of this synthetic clip, not a bug: there is no way to recover true turn
speed from discrete move/view labels alone.

Output: examples/racer/actions.npy, shape [num_frames, 17], float32.
num_frames is the largest value <= len(0001.json) satisfying H3's
(num_frames - 5) % 17 == 0 constraint (see abot_action.py's latent_t_for) --
printed so it can be passed to infer.py's --num-frames.

Run: python3 examples/racer/convert_actions.py
"""
import json
import sys
from pathlib import Path

import numpy as np

CODE_ABOT = Path(__file__).resolve().parents[2] / "code" / "abot"
sys.path.insert(0, str(CODE_ABOT))
import abot_action as A  # noqa: E402

racer_dir = Path(__file__).parent
actions = json.loads((racer_dir / "0001.json").read_text())

# Largest num_frames <= len(actions) satisfying (n - 5) % 17 == 0.
n = len(actions)
n = n - ((n - 5) % 17)
if n < 5:
    raise ValueError(f"0001.json has too few ticks ({len(actions)}) for even one valid --num-frames")

mat = np.zeros((n, A.ACTION_DIM), dtype=np.float32)
w_idx = A.KEY_COLS.index("W")
j_idx = A.KEY_COLS.index("J")
l_idx = A.KEY_COLS.index("L")

for i, a in enumerate(actions[:n]):
    if a["move"] == "go forward":
        mat[i, w_idx] = 1.0
    if a["view"] == "turn left":
        mat[i, j_idx] = 1.0
    elif a["view"] == "turn right":
        mat[i, l_idx] = 1.0

out_path = racer_dir / "actions.npy"
np.save(out_path, mat)
print(f"Wrote {out_path}: shape {mat.shape}")
print(f"Used {n}/{len(actions)} ticks from 0001.json")
print(f"--num-frames {n}")
