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
num_frames is the largest value <= min(len(0001.json), --max-frames)
satisfying H3's (num_frames - 5) % 17 == 0 constraint (see
abot_action.py's latent_t_for) -- printed so it can be passed to infer.py's
--num-frames.

--max-frames matters for real: CONFIRMED on real hardware that the full
1382-frame clip OOMs even on a 249 GiB GPU -- the DiT's action-block-mask
construction (_build_action_block_masks -> create_block_mask) allocates a
dense [length, length] tensor that scales with sequence length, tried to
allocate 209.81 GiB for the full clip.

Default here is 243, not the README-matching 124: 0001.json's first real
"turn right" segment is ticks 81-161 (81 ticks) -- 124 frames cuts it off
after only 43 of those ticks; 175 is the smallest valid (17k+5) value that
captures the complete first turn, and ticks 0-174 were then edited (see
below) to be "turn right" throughout that window. 243 extends a bit
further (~10.1s clip) and happens to land exactly at the start of the
next genuine "turn right" segment in the original recording (243-323),
so ticks 175-242 are unedited (no-op, as originally recorded) before real
turning resumes right where this window ends. Memory cost of the longer
mask is still tiny relative to the full-clip OOM (roughly (72/407)^2 of
the 209.81 GiB that failed, using latent_t as the scaling proxy -- a few
GB, not hundreds).

NOTE: 0001.json itself was edited (ticks 0-174's "view" field, originally
mostly "no-op", changed to "turn right") to make the used window turn
right throughout instead of only ticks 81-161 -- this script just converts
whatever's currently in 0001.json, it doesn't know that edit happened.

Run: python3 examples/racer/convert_actions.py [--max-frames N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_ABOT = Path(__file__).resolve().parents[2] / "code" / "abot"
sys.path.insert(0, str(CODE_ABOT))
import abot_action as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--max-frames", type=int, default=243,
                help="cap on num_frames, must be 17k+5 -- see module docstring for why "
                     "243 (~10.1s, not the full clip, not the README's 124) is the default")
args = ap.parse_args()
if (args.max_frames - 5) % 17:
    ap.error(f"--max-frames must be 17k+5 (124, 243, 481, ...), got {args.max_frames}")

racer_dir = Path(__file__).parent
actions = json.loads((racer_dir / "0001.json").read_text())

# Largest num_frames <= min(len(actions), args.max_frames) satisfying (n - 5) % 17 == 0.
n = min(len(actions), args.max_frames)
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
