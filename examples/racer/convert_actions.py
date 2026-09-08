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

Default here is 481, not the README-matching 124: this is a longer (~20s)
window meant to drive the buggy into the wall/fence per prompt.txt's
collision clause and hold the turn the whole way, rather than cutting off
mid-turn. Memory cost of the longer mask is still small relative to the
full-clip OOM (roughly (481/1382)^2 of the 209.81 GiB that failed, using
latent_t as the scaling proxy -- tens of GB, not hundreds).

NOTE: 0001.json itself was edited (ticks 0-480's "view" field set to
"turn left" throughout, since prompt.txt puts the wall/building on the
left and the fence on the right) -- this script just converts whatever's
currently in 0001.json, it doesn't know that edit happened.

Also usable on straight.json / left.json / right.json (same 481-tick window as
0001.json, "move" unchanged, "view" held constant at no-op/turn left/turn
right respectively) via --actions-file, to get three actually-comparable
clips instead of editing 0001.json in place each time.

Run: python3 examples/racer/convert_actions.py [--max-frames N] [--actions-file NAME.json]
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
ap.add_argument("--max-frames", type=int, default=481,
                help="cap on num_frames, must be 17k+5 -- see module docstring for why "
                     "481 (~20s, not the full clip, not the README's 124) is the default")
ap.add_argument("--actions-file", default="0001.json",
                help="input JSON in examples/racer/, e.g. straight.json/left.json/right.json "
                     "(default 0001.json)")
ap.add_argument("--out", default=None,
                help="output .npy path (default: <actions-file stem>_actions.npy for a "
                     "non-default --actions-file, else actions.npy)")
args = ap.parse_args()
if (args.max_frames - 5) % 17:
    ap.error(f"--max-frames must be 17k+5 (124, 243, 481, ...), got {args.max_frames}")

racer_dir = Path(__file__).parent
in_path = racer_dir / args.actions_file
actions = json.loads(in_path.read_text())

# Largest num_frames <= min(len(actions), args.max_frames) satisfying (n - 5) % 17 == 0.
n = min(len(actions), args.max_frames)
n = n - ((n - 5) % 17)
if n < 5:
    raise ValueError(f"{in_path.name} has too few ticks ({len(actions)}) for even one valid --num-frames")

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

if args.out:
    out_path = racer_dir / args.out
elif args.actions_file == "0001.json":
    out_path = racer_dir / "actions.npy"
else:
    out_path = racer_dir / f"{Path(args.actions_file).stem}_actions.npy"
np.save(out_path, mat)
print(f"Wrote {out_path}: shape {mat.shape}")
print(f"Used {n}/{len(actions)} ticks from {in_path.name}")
print(f"--num-frames {n}")
