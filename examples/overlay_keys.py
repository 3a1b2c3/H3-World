#!/usr/bin/env python3
"""Burn a WASDIJKL key-press overlay onto a generated H3-World video, synced
frame-for-frame against the action matrix that drove the generation.

Row i of the action matrix maps 1:1 to output video frame i -- no
resampling needed. This isn't an assumption; it's how the pipeline is
built: abot_action.py's own docstring says video and action are cut with
the same explicit frame-index list at slicing time, so they're
frame-aligned by construction (verified there: zero duplicate frames).

Draws all 8 of abot_action.ACTIVE_KEY_COLS (W, A, S, D, I, J, K, L) as a
small on-screen keyboard, highlighting whichever are set (> 0) on that
frame. Works for any --action-file matrix, not just the racer example --
for a constant --action-preset run, pass a matrix built the same way
infer.py builds `keys9` (see build_keys9_from_preset() below) instead.

Run:
    python3 examples/overlay_keys.py \\
        --video outputs/example_racer.mp4 \\
        --actions examples/racer/actions.npy \\
        --out outputs/example_racer_overlay.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

CODE_ABOT = Path(__file__).resolve().parent.parent / "code" / "abot"
sys.path.insert(0, str(CODE_ABOT))
import abot_action as A  # noqa: E402

# 3x3 grid layout: WASD on the left (movement), IJKL on the right (camera) --
# matches the physical keyboard layout these correspond to.
KEY_LAYOUT = {
    "W": (1, 0), "A": (0, 1), "S": (1, 1), "D": (2, 1),
    "I": (5, 0), "J": (4, 1), "K": (5, 1), "L": (6, 1),
}
KEY_SIZE = 28
KEY_GAP = 4
ORIGIN = (16, 16)
COLOR_OFF = (60, 60, 60, 180)
COLOR_ON = (255, 210, 0, 220)
COLOR_TEXT_OFF = (200, 200, 200, 255)
COLOR_TEXT_ON = (20, 20, 20, 255)


def draw_keys_frame(frame: np.ndarray, active_keys: set[str]) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for key, (col, row) in KEY_LAYOUT.items():
        x = ORIGIN[0] + col * (KEY_SIZE + KEY_GAP)
        y = ORIGIN[1] + row * (KEY_SIZE + KEY_GAP)
        on = key in active_keys
        fill = COLOR_ON if on else COLOR_OFF
        text_color = COLOR_TEXT_ON if on else COLOR_TEXT_OFF
        draw.rectangle([x, y, x + KEY_SIZE, y + KEY_SIZE], fill=fill, outline=(0, 0, 0, 255))
        # Centered text without needing a specific font file -- default PIL
        # bitmap font is small but always available, no extra dependency.
        bbox = draw.textbbox((0, 0), key)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (KEY_SIZE - tw) / 2, y + (KEY_SIZE - th) / 2 - bbox[1]), key, fill=text_color)
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return np.asarray(composited)


def build_keys9_from_preset(preset_keys: tuple[str, ...], num_frames: int) -> np.ndarray:
    """Matches infer.py's constant-preset keys9 construction, for overlaying
    a --action-preset run instead of a --action-file one -- same key set
    held for every frame."""
    mat = np.zeros((num_frames, A.ACTION_DIM), dtype=np.float32)
    for key in preset_keys:
        if key in A.KEY_COLS:
            mat[:, A.KEY_COLS.index(key)] = 1.0
    return mat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--actions", required=True, type=Path,
                     help="[num_frames, 17] action matrix, e.g. examples/racer/actions.npy "
                          "(same file passed to infer.py's --action-file)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    mat = np.load(args.actions)
    frames = iio.imread(args.video, plugin="pyav")
    meta = iio.immeta(args.video, plugin="pyav")
    fps = meta.get("fps", 24)

    if len(frames) != mat.shape[0]:
        print(f"WARNING: video has {len(frames)} frames but actions matrix has {mat.shape[0]} rows -- "
              f"using min({len(frames)}, {mat.shape[0]}); this shouldn't happen for a video/actions pair "
              f"that came from the same infer.py run.", file=sys.stderr)
    n = min(len(frames), mat.shape[0])

    out_frames = []
    for i in range(n):
        active = {key for key in A.ACTIVE_KEY_COLS if mat[i, A.KEY_COLS.index(key)] > 0}
        out_frames.append(draw_keys_frame(frames[i], active))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.stack(out_frames), fps=fps, plugin="pyav", codec="libx264")
    print(f"Wrote {args.out}: {n} frames @ {fps}fps")


if __name__ == "__main__":
    main()
