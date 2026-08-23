#!/usr/bin/env python3
"""Render the README demo reel: three chapters, each with the action prompt on screen.

    1. Generated vs ground truth        — does the model follow the real action track?
    2. Same first frame, new actions    — swap only the keys, watch the motion change
    3. Longer videos                    — one-shot 10s and chunked 15.5s

Every chapter shows the per-latent action text that drove the frame, plus the 9-key
HUD (same layout as viz_action.py's raw-data HUD) and a 37-cell timeline cursor.
Text is English throughout so the reel matches the README.

Usage:
    python3 code/abot/build_demo_video.py --out docs/assets/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abot_action as A  # noqa: E402
import action_script as S  # noqa: E402
import infer_abot as I  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data/clips"
META = ROOT / "data/abot_meta_test_128.jsonl"
INFER = ROOT / "output/abot_inference"
FFMPEG = (ROOT / "../envs/minimax_h3/lib/python3.10/site-packages/imageio_ffmpeg/"
          "binaries/ffmpeg-linux-x86_64-v7.0.2").resolve()
SID = "a3ad9c24bda131dfa0ea18efe44a4e8b"

W, H, FPS = 1280, 720, 24
BG, INK, DIM, FAINT = (11, 14, 19), (234, 238, 242), (140, 150, 162), (39, 49, 61)
ACCENT, WARM = (62, 207, 182), (230, 138, 60)

_M = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono%s.ttf"
_S = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans%s.ttf"
F_TITLE = ImageFont.truetype(_S % "-Bold", 34)
F_SUB = ImageFont.truetype(_S % "", 19)
F_LBL = ImageFont.truetype(_M % "-Bold", 15)
F_TXT = ImageFont.truetype(_S % "", 22)
F_SM = ImageFont.truetype(_M % "", 13)
F_XS = ImageFont.truetype(_M % "", 11)

KEYPAD = [("MOVE", [[None, "W", None], ["A", "S", "D"]]),
          ("VIEW", [[None, "I", None], ["J", "K", "L"]])]


def read_frames(path: Path, size: tuple[int, int], limit: int | None = None):
    import av
    out = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            out.append(f.to_image().convert("RGB").resize(size, Image.LANCZOS))
            if limit and len(out) >= limit:
                break
    return out


def keys_for(n_frames: int, preset=None, seed=None, tile_to=None, sid=None):
    """Per-latent 9-bit key matrix + the sentence each latent carries."""
    I.set_frames(n_frames)
    if seed is not None:
        k9 = I.random_keys9(seed, I.LATENT_T)
    elif preset is not None:
        k9 = I.preset_keys9(preset)
    else:
        want = sid or SID
        row = next(json.loads(l) for l in open(META)
                   if json.loads(l)["sample_id"] == want)
        _, _, k9, _ = I.load_action(row, CLIPS)
    if tile_to and k9.shape[0] != tile_to:
        k9 = I.tile_keys9(k9, tile_to)
    return k9, S.annotate_from_keys9(k9)


def latent_at(frame_idx: int, latent_t: int) -> int:
    spans = [e - s for s, e in A.frame_spans(latent_t)]
    acc, k = 0, 0
    for i, n in enumerate(spans):
        if frame_idx < acc + n:
            return i
        acc += n
        k = i
    return k


def draw_pad(d: ImageDraw.ImageDraw, x: int, y: int, k9, cell=26, gap=4):
    """Two 3-wide keypads + a FAST chip; pressed = filled. Returns width used."""
    kidx = {n: i for i, n in enumerate(S.KEYS9)}
    x0 = x
    for title, rows in KEYPAD:
        d.text((x0, y - 15), title, font=F_XS, fill=DIM)
        for r, row in enumerate(rows):
            for c, name in enumerate(row):
                if name is None:
                    continue
                bx, by = x0 + c * (cell + gap), y + r * (cell + gap)
                on = bool(k9[kidx[name]])
                d.rounded_rectangle([bx, by, bx + cell, by + cell], 5,
                                    fill=ACCENT if on else None,
                                    outline=ACCENT if on else FAINT, width=2)
                d.text((bx + cell / 2, by + cell / 2), name, font=F_LBL,
                       fill=(11, 14, 19) if on else DIM, anchor="mm")
        x0 += 3 * (cell + gap) + 22
    on = bool(k9[kidx["F"]])
    d.text((x0, y - 15), "FAST", font=F_XS, fill=DIM)
    d.rounded_rectangle([x0, y, x0 + 74, y + cell], 5, fill=WARM if on else None,
                        outline=WARM if on else FAINT, width=2)
    d.text((x0 + 37, y + cell / 2), "FAST", font=F_SM,
           fill=(11, 14, 19) if on else DIM, anchor="mm")
    return x0 + 74 - x


def draw_timeline(d, x, y, w, latent_t, cur, k9all, marks=()):
    """One cell per latent; pressed latents tinted, current one boxed."""
    gap = 1
    cw = (w - gap * (latent_t - 1)) / latent_t
    for k in range(latent_t):
        bx = x + k * (cw + gap)
        active = bool(k9all[k][:8].any())
        d.rectangle([bx, y, bx + cw, y + 14],
                    fill=(46, 92, 84) if active else (30, 38, 47))
        if k == cur:
            d.rectangle([bx - 1, y - 3, bx + cw + 1, y + 17], outline=ACCENT, width=2)
    for m in marks:
        mx = x + m * (cw + gap)
        d.line([mx, y - 6, mx, y + 20], fill=WARM, width=2)


def panel(d, y, k9, text, latent_t, cur, k9all, marks=(), note=None):
    """The shared bottom panel: keypad + sentence + timeline."""
    used = draw_pad(d, 40, y + 16, k9)
    tx = 40 + used + 40
    d.text((tx, y + 8), "ACTION PROMPT", font=F_XS, fill=DIM)
    d.text((tx, y + 26), text, font=F_TXT, fill=INK)
    if note:
        d.text((tx, y + 56), note, font=F_SM, fill=DIM)
    draw_timeline(d, 40, y + 92, W - 80, latent_t, cur, k9all, marks)
    d.text((40, y + 112), "latent 0", font=F_XS, fill=DIM)
    d.text((W - 40, y + 112), f"latent {latent_t - 1}", font=F_XS, fill=DIM, anchor="ra")


def header(d, title, sub):
    d.text((40, 30), title, font=F_TITLE, fill=INK)
    d.text((40, 74), sub, font=F_SUB, fill=DIM)
    d.line([40, 108, W - 40, 108], fill=FAINT, width=1)


def title_card(text, sub, n=36):
    """A short interstitial so chapters do not run together."""
    out = []
    for i in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        a = min(1.0, i / 8) * min(1.0, (n - i) / 8)
        col = tuple(int(11 + (234 - 11) * a) for _ in range(3))
        d.text((W / 2, H / 2 - 26), text, font=F_TITLE, fill=col, anchor="mm")
        d.text((W / 2, H / 2 + 24), sub, font=F_SUB,
               fill=tuple(int(11 + (140 - 11) * a) for _ in range(3)), anchor="mm")
        out.append(im)
    return out


def chapter_pair(left, right, lname, rname, k9, script, latent_t,
                 title, sub, marks=(), note=None):
    """Two videos side by side sharing one action-prompt panel."""
    # 只有一个视频时画面可以大一些；成对时受画布宽度限制
    vw, vh = (596, 344) if left else (740, 427)
    L = read_frames(left, (vw, vh)) if left else None
    R = read_frames(right, (vw, vh))
    n = min(len(R), len(L)) if L else len(R)
    frames = []
    for i in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        header(d, title, sub)
        if L:
            im.paste(L[i], (40, 132))
            d.text((40, 122), lname, font=F_LBL, fill=DIM, anchor="ls")
            im.paste(R[i], (W - 40 - vw, 132))
            d.text((W - 40 - vw, 122), rname, font=F_LBL, fill=ACCENT, anchor="ls")
        else:
            im.paste(R[i], ((W - vw) // 2, 132))
            d.text(((W - vw) // 2, 122), rname, font=F_LBL, fill=ACCENT, anchor="ls")
        k = latent_at(i, latent_t)
        panel(d, 512 if L else 578, k9[k], script[k], latent_t, k, k9, marks, note)
        frames.append(im)
    return frames


def chapter_quad(items, title, sub):
    """Four presets at once: same first frame, four different action prompts."""
    # 高度是硬约束：header 约 108，两行各 (vh + 46) 的标签块，总和必须 <= H。
    vw, vh, lab, gap = 442, 255, 46, 36
    x0 = (W - (2 * vw + gap)) // 2
    y0 = 116
    vids = [(read_frames(p, (vw, vh)), name, text) for p, name, text in items]
    n = min(len(v[0]) for v in vids)
    pos = [(x0, y0), (x0 + vw + gap, y0),
           (x0, y0 + vh + lab), (x0 + vw + gap, y0 + vh + lab)]
    frames = []
    for i in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((40, 26), title, font=F_TITLE, fill=INK)
        d.text((40, 68), sub, font=F_SUB, fill=DIM)
        for (fr, name, text), (x, y) in zip(vids, pos):
            im.paste(fr[i], (x, y))
            d.text((x, y + vh + 6), name, font=F_LBL, fill=ACCENT)
            d.text((x, y + vh + 26), text, font=F_SM, fill=DIM)
        frames.append(im)
    return frames


def write(frames, out: Path, crf: int = 28):
    """crf 28 keeps the reel under 10 MB so it can live in the repo.
    Lower (better) values blow past that: 23 lands around 20 MB for 56 s."""
    out.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        [str(FFMPEG), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    for im in frames:
        p.stdin.write(im.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/assets/demo.mp4")
    ap.add_argument("--crf", type=int, default=28)
    args = ap.parse_args()
    frames: list[Image.Image] = []

    # ── 1. generated vs ground truth ────────────────────────────────────────
    print("  chapter 1: generated vs GT …", flush=True)
    frames += title_card("Action-conditioned generation",
                         "one English sentence per latent frame, 37 per clip")
    # 三条不同的测试样本，而不是只看一条 —— 一条容易挑到最好的那个
    for n, gpu in enumerate((0, 3, 5)):
        run = INFER / f"step9840_text_gpu{gpu}"
        gen = next(run.glob("*/generated.mp4"), None)
        gt = next(run.glob("*/gt.mp4"), None)
        if not gen or not gt:
            continue
        sid = gen.parent.name.rsplit("_w", 1)[0]
        k9, sc = keys_for(124, sid=sid)
        print(f"    sample {n + 1}/3  {sid[:8]}", flush=True)
        frames += chapter_pair(gt, gen, "GROUND TRUTH", "GENERATED", k9, sc, 37,
                               f"Generated vs ground truth  ({n + 1}/3)",
                               f"held-out clip {sid[:8]} — its own recorded key presses")

    # ── 2. same first frame, different action prompts ───────────────────────
    print("  chapter 2: swap the actions …", flush=True)
    frames += title_card("Same first frame, new actions",
                         "eight key presets, only the keys change")
    # 一屏放不下 8 个，就在时间上分两批。分组按语义：先移动，后相机。
    batches = [("forward", "back", "strafe-left", "strafe-right"),
               ("still", "pan-left", "pan-right", "pan-right-fast")]
    labels = ["movement keys — W / S / A / D",
              "camera keys — none / J / L / L+FAST"]
    for bi, (names, lab) in enumerate(zip(batches, labels)):
        quad = []
        for name in names:
            g = next((INFER / f"ab8_step9840_{name}").glob("*/generated.mp4"), None)
            if g is None:
                continue
            _, s1 = keys_for(124, preset=name)
            quad.append((g, name, s1[0]))
        if quad:
            frames += chapter_quad(
                quad, f"Same first frame, eight action prompts  ({bi + 1}/2)",
                f"identical seed and scene — {lab}")

    # ── 3. longer videos ────────────────────────────────────────────────────
    print("  chapter 3: longer videos …", flush=True)
    frames += title_card("Longer videos", "no retraining, no architecture change")
    k9b, scb = keys_for(243)
    long1 = next((INFER / "long10s_step9840").glob("*/generated.mp4"))
    frames += chapter_pair(None, long1, "", "ONE SHOT · 10.1s", k9b, scb, 72,
                           "One-shot 10 seconds",
                           "num_frames 124 -> 243; the clip's keys tiled to 72 latents")

    joined = INFER / "chunked_step9840/joined.mp4"
    if joined.is_file():
        I.set_frames(124)
        k9c = np.concatenate([I.random_keys9(c) for c in range(3)], axis=0)
        scc = []
        for c in range(3):
            scc += S.annotate_from_keys9(I.random_keys9(c))
        frames += chapter_pair(None, joined, "", "CHUNKED · 15.5s", k9c, scc, 111,
                               "Chunked continuation, 3 x 5.2 seconds",
                               "each chunk starts from the previous chunk's last frame",
                               marks=(37, 74), note="orange marks = chunk seams")

    print(f"  encoding {len(frames)} frames ({len(frames)/FPS:.1f}s) …", flush=True)
    write(frames, args.out, args.crf)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 2**20:.1f} MB, "
          f"{len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
