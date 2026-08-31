#!/usr/bin/env python3
"""ABot-World-Explorer-500h raw data -> training clips + metadata for MiniMax-H3.

One episode is 60s / 1800 frames / 1920x1080 / 30fps / 111 MB, while H3 only
takes 124 frames @24fps per training sample. So the raw data has to be
sliced first, and slicing does three things at once:

  1. **30fps -> 24fps**: drop the 5th of every 5 source frames (an exact
     24fps, precisely expressible with ffmpeg's select filter). The cut
     clips are already 24fps, so stage 1's `LoadVideo(fix_frame_rate=True,
     frame_rate=24)` reads them back as the **identity mapping** -- verified:
     zero duplicate frames, no dropped tail frames.
     Compare V-Rising (16fps source upsampled to 24fps): 35/107 = 32.7% of
     frames are duplicates, and 0.56s is dropped off the tail.
  2. **1920x1080 -> HEIGHTxWIDTH**: scale to the target height, then center
     crop. The cut clips are already at training resolution, so stage 1's
     ImageCropAndResize becomes a no-op instead of resampling a second time.
  3. **Action alignment**: the video frames and the action array are read
     using the **same explicit list of source frame indices**, so they are
     frame-aligned by construction, rather than relying on inferring H3's
     internal round() mapping. This closes the "temporal alignment not yet
     verified" risk noted in the world-model docs, since alignment is now
     true by construction and can be spot-checked with --verify.

Row-order discipline: every tier is a prefix of the same global ordering, so
the cache built for a small tier can be reused directly by a larger tier.
When scaling up, only append -- never reorder the prefix.

Usage:
    python3 build_abot_clips.py --plan                      # print the plan only, no writes
    python3 build_abot_clips.py --num-clips 64   --workers 16
    python3 build_abot_clips.py --num-clips 8000 --workers 24
    python3 build_abot_clips.py --verify 8                  # spot-check alignment
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abot_action as A

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = os.environ.get("ABOT_SRC_ROOT")
if not SRC_ROOT:
    raise SystemExit(
        "Set ABOT_SRC_ROOT to the local path of the downloaded "
        "ABot-World-Explorer-500h dataset before running this script."
    )
OUT_ROOT = os.environ.get("ABOT_OUT_ROOT", str(PROJECT_ROOT / "data"))
CLIP_DIR = os.environ.get("ABOT_CLIP_DIR", f"{OUT_ROOT}/clips")
MANIFEST = os.environ.get("ABOT_MANIFEST", f"{OUT_ROOT}/abot_manifest.jsonl")

NUM_FRAMES = 124          # 17*7+5, H3's native value (every official example uses 124)
PAD = 6                   # write a few extra frames so floor(duration*24) never dips below 124
N_OUT = NUM_FRAMES + PAD  # each clip actually writes 130 frames
# Training resolution. Both dimensions must be a multiple of 32 -- the video
# VAE compresses space 16x and the DiT patch embed another 2x2, so a
# non-multiple makes patchify_video's reshape raise RuntimeError (which is
# exactly what happens at 720p). 832x480 is DiffSynth's example default;
# 1344x768 is H3's own native resolution.
HEIGHT = int(os.environ.get("ABOT_HEIGHT", 480))
WIDTH = int(os.environ.get("ABOT_WIDTH", 832))
SEED = 20260817


def check_resolution(h: int, w: int) -> None:
    bad = [f"{n}={v}" for n, v in (("height", h), ("width", w)) if v % 32]
    if bad:
        raise SystemExit(f"height and width must both be multiples of 32 (VAE 16x * patch 2x2), got: {', '.join(bad)}")


def scaled_width(src_w: int, src_h: int, height: int = None) -> int:
    """Reproduces ffmpeg `scale=-2:HEIGHT`: scale to the target height, round the width to the nearest even number."""
    height = HEIGHT if height is None else height
    return int(round(src_w * height / src_h / 2)) * 2


def resolve_ffmpeg() -> str:
    configured = os.environ.get("ABOT_FFMPEG")
    if configured:
        return configured
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("ffmpeg not found; set ABOT_FFMPEG to its executable path") from exc


FFMPEG = resolve_ffmpeg()


# --------------------------------------------------------------------------- #
def episode_order():
    """Global, fixed episode order. Depends only on sample_id and SEED, so it's reproducible on any machine."""
    ids = []
    for prefix in sorted(os.listdir(f"{SRC_ROOT}/data")):
        for sid in sorted(os.listdir(f"{SRC_ROOT}/data/{prefix}")):
            ids.append(sid)
    random.Random(SEED).shuffle(ids)
    return ids


def clip_plan(n_clips: int, ids):
    """Global order of (sample_id, window_index): window 0 for every episode
    first, then window 1 for every episode, and so on. This way both "add
    more windows" and "add more episodes" are pure appends, and the prefix
    stays stable."""
    plan, w = [], 0
    while len(plan) < n_clips:
        for sid in ids:
            plan.append((sid, w))
            if len(plan) >= n_clips:
                break
        w += 1
    return plan


def window_start(sid: str, w: int, total_frames: int, span: int) -> int:
    """Source-frame start of window w.

    The episode is split into n_slots non-overlapping slots, the slot order
    is shuffled with sample_id as the seed, and window w takes the w-th
    slot of the shuffled order. This does two things:
      * multiple windows from the same episode never overlap (otherwise
        adjacent windows would be near-duplicate samples);
      * **window 0 is spread evenly across the full 60s**, instead of every
        episode being cut from the start -- the start of a clip is often a
        camera-settling, action-sparse stretch, and always cutting from
        there would skew the action distribution.
    """
    usable = total_frames - span
    if usable < 0:
        raise ValueError(f"episode only has {total_frames} frames, can't fit a {span}-frame window")
    n_slots = max(1, total_frames // span)
    rng = random.Random(f"{sid}:{SEED}")
    slots = list(range(n_slots))
    rng.shuffle(slots)
    slot = slots[w % n_slots]
    jitter = random.Random(f"{sid}:{w}:jit").randrange(0, max(1, span // 2))
    return min(usable, slot * span + jitter)


# --------------------------------------------------------------------------- #
MIN_PROMPT_WORDS = 30


def pick_prompt(cap: dict) -> str:
    """Choose a scene prompt. Prefers `scene_static` (window-independent),
    falls back to `narrative` when it's too short.

    The source captions do contain some bad data: 1 of 8000 clips has
    `scene_static` literally equal to `'!!!'`. It's not worth dropping the
    row for this (dropping a row shifts every later row's index, which
    invalidates the whole cache), but it's also not worth training on
    `'!!!'` -- falling back to `narrative` is free.
    `narrative` describes motion over the full 60s and doesn't match a
    single window, so it's only used as a fallback.
    """
    s = (cap.get("scene_static") or "").strip()
    if len(s.split()) >= MIN_PROMPT_WORDS:
        return s
    return (cap.get("narrative") or "").strip() or s


def build_one(task):
    sid, w, crf = task
    prefix = sid[:2]
    ep_dir = f"{SRC_ROOT}/data/{prefix}/{sid}"
    out_dir = f"{CLIP_DIR}/{prefix}"
    stem = f"{sid}_w{w:03d}"
    mp4 = f"{out_dir}/{stem}.mp4"
    npy = f"{out_dir}/{stem}.npy"
    try:
        ep = A.read_episode(f"{ep_dir}/annotations.tar")
        if ep["control_scheme"] != "WASD_QE_locomotion_IJKL_rotation":
            return dict(sid=sid, w=w, ok=False, err=f"control_scheme={ep['control_scheme']}")
        span = A.window_span(N_OUT)
        s = window_start(sid, w, ep["total_frames"], span)

        os.makedirs(out_dir, exist_ok=True)
        if not (os.path.exists(mp4) and os.path.exists(npy)):
            scale = A.episode_translation_scale(ep)
            act = A.window_action_matrix(ep, s, N_OUT, scale)
            np.save(npy + ".tmp.npy", act)
            os.replace(npy + ".tmp.npy", npy)

            vf = (f"select='between(n\\,{s}\\,{s + span - 1})"
                  f"*not(eq(mod(n-{s}\\,5)\\,4))'"
                  f",setpts=N/24/TB,scale=-2:{HEIGHT},crop={WIDTH}:{HEIGHT}")
            tmp = mp4 + ".tmp.mp4"
            subprocess.run(
                [FFMPEG, "-y", "-v", "error", "-i", f"{ep_dir}/video.mp4",
                 "-vf", vf, "-r", "24", "-frames:v", str(N_OUT),
                 "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
                 "-pix_fmt", "yuvj420p", "-an", tmp],
                check=True, capture_output=True)
            os.replace(tmp, mp4)          # same-directory rename is atomic; if the file exists, it's complete

        cap = ep["caption"]
        return dict(sid=sid, w=w, ok=True, src_start=s,
                    video=f"{prefix}/{stem}.mp4", action=f"{prefix}/{stem}.npy",
                    prompt=pick_prompt(cap),
                    narrative=cap.get("narrative", "").strip(),
                    perspective=cap.get("perspective"))
    except subprocess.CalledProcessError as e:
        return dict(sid=sid, w=w, ok=False, err=f"ffmpeg: {e.stderr.decode()[:200]}")
    except Exception as e:                                    # noqa: BLE001
        return dict(sid=sid, w=w, ok=False, err=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
def verify(n: int, shifts=(-2, -1, 0, 1, 2)):
    """Spot-check that the clips are frame-aligned with the source video.

    The check uses **the argmin of a shift sweep**, not an absolute pixel
    difference. Reason: the reference frames here reproduce ffmpeg's
    bicubic scaling with PIL's bilinear resize, which already has a floor
    noise of about 5-6/255, and adjacent-frame differences in a slow
    passage can be the same order of magnitude -- comparing "aligned vs.
    off by one frame" on the absolute value alone wouldn't be conclusive.
    But that floor noise is **common to every shift**, so the argmin still
    lands cleanly on 0.
    """
    import imageio.v2 as imageio
    import PIL.Image
    sys.path.insert(0, os.environ.get("DIFFSYNTH_ROOT", str(PROJECT_ROOT / "DiffSynth-Studio-h3-v2")))
    from diffsynth.core.data.operators import LoadVideo

    rows = [json.loads(l) for l in open(MANIFEST)]
    picks = random.Random(0).sample(rows, min(n, len(rows)))
    offs = A.window_offsets(N_OUT)
    probes = [10, 40, 80, 118]
    lv = LoadVideo(num_frames=NUM_FRAMES, time_division_factor=17,
                   time_division_remainder=5, frame_rate=24, fix_frame_rate=True)
    print(f"{'sample':14s} {'nfrm':>5s} {'dup':>4s}  " +
          "  ".join(f"shift{k:+d}" for k in shifts) + "   argmin  verdict")
    n_pass = 0
    for r in picks:
        sid, s = r["sample_id"], r["src_start"]
        frames = [np.asarray(f).astype(np.float32) for f in lv(f"{CLIP_DIR}/{r['video']}")]
        dup = sum(1 for a, b in zip(frames, frames[1:]) if np.array_equal(a, b))

        want = {s + offs[j + k] for j in probes for k in shifts}
        got, hi = {}, max(want)
        rd = imageio.get_reader(f"{SRC_ROOT}/data/{sid[:2]}/{sid}/video.mp4")
        for i, fr in enumerate(rd):
            if i in want:
                sw = scaled_width(fr.shape[1], fr.shape[0])
                off = (sw - WIDTH) // 2                      # crop=WIDTH:HEIGHT is a center crop
                im = PIL.Image.fromarray(fr).resize((sw, HEIGHT), PIL.Image.BILINEAR)
                got[i] = np.asarray(im)[:, off:off + WIDTH].astype(np.float32)
            if i >= hi:
                break
        rd.close()

        scores = [float(np.mean([np.abs(got[s + offs[j + k]] - frames[j]).mean()
                                 for j in probes])) for k in shifts]
        amin = shifts[int(np.argmin(scores))]
        ok = len(frames) == NUM_FRAMES and dup == 0 and amin == 0
        n_pass += ok
        print(f"{sid[:12]:14s} {len(frames):5d} {dup:4d}  " +
              "  ".join(f"{v:6.2f}" for v in scores) +
              f"   {amin:+d}     {'PASS' if ok else 'FAIL'}")
    print(f"\n{n_pass}/{len(picks)} passed (124 frames / 0 duplicate frames / shift-sweep argmin=0)")


# --------------------------------------------------------------------------- #
def main():
    # Set globals before the process pool is created, so forked workers inherit them directly.
    global HEIGHT, WIDTH, CLIP_DIR, MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--tiers", default="64,2000,8000",
                    help="additional prefix tiers to export, comma-separated")
    ap.add_argument("--plan", action="store_true", help="print the plan only")
    ap.add_argument("--verify", type=int, default=0, help="spot-check N clips' alignment, then exit")
    ap.add_argument("--height", type=int, default=HEIGHT, help="training height, must be a multiple of 32")
    ap.add_argument("--width", type=int, default=WIDTH, help="training width, must be a multiple of 32")
    ap.add_argument("--clip-dir", default=CLIP_DIR, help="clip output directory; use a new directory when changing resolution")
    ap.add_argument("--manifest", default=MANIFEST, help="manifest output path")
    args = ap.parse_args()

    HEIGHT, WIDTH, CLIP_DIR, MANIFEST = args.height, args.width, args.clip_dir, args.manifest
    check_resolution(HEIGHT, WIDTH)

    if args.verify:
        verify(args.verify)
        return

    span = A.window_span(N_OUT)
    ids = episode_order()
    plan = clip_plan(args.num_clips, ids)
    print(f"source: {SRC_ROOT}  ({len(ids)} episodes)")
    print(f"output: {CLIP_DIR}")
    print(f"clip: {N_OUT} frames written / first {NUM_FRAMES} frames used for training @24fps "
          f"({NUM_FRAMES / 24:.3f}s), {WIDTH}x{HEIGHT}, {span} source frames consumed per clip")
    print(f"latent_t = {A.latent_t_for(NUM_FRAMES)}   action_dim = {A.ACTION_DIM}")
    print(f"planned {len(plan)} clips, max window index {max(w for _, w in plan)}, workers={args.workers}")
    if args.plan:
        for sid, w in plan[:5]:
            print(f"   {sid} w{w:03d}")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    t0 = time.time()
    results = {}
    tasks = [(sid, w, args.crf) for sid, w in plan]
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(build_one, t): (t[0], t[1]) for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results[(r["sid"], r["w"])] = r
            if i % 50 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  {i}/{len(tasks)}  {el:.0f}s  {el / i:.2f}s/clip  "
                      f"ETA {(len(tasks) - i) * el / i / 60:.1f}min", flush=True)

    ok = [results[k] for k in plan if results[k]["ok"]]
    bad = [results[k] for k in plan if not results[k]["ok"]]
    man = MANIFEST
    os.makedirs(os.path.dirname(man) or ".", exist_ok=True)
    with open(man, "w") as f:
        for r in ok:
            f.write(json.dumps({
                "video": r["video"], "input_audio": r["video"], "prompt": r["prompt"],
                "action": r["action"], "sample_id": r["sid"], "window": r["w"],
                "src_start": r["src_start"], "perspective": r["perspective"],
                "narrative": r["narrative"],
            }, ensure_ascii=False) + "\n")
    print(f"\nsucceeded {len(ok)}  failed {len(bad)}  ->  {man}")
    for r in bad[:10]:
        print(f"   FAIL {r['sid']} w{r['w']}: {r['err']}")

    lines = open(man).read().splitlines()
    for tier in [int(x) for x in args.tiers.split(",") if x]:
        if tier > len(lines):
            print(f"   tier {tier} exceeds the {len(lines)} successful clips, skipping")
            continue
        p = f"{os.path.dirname(MANIFEST) or '.'}/abot_meta_{tier}.jsonl"
        with open(p, "w") as f:
            f.write("\n".join(lines[:tier]) + "\n")
        print(f"   tier {tier:6d} -> {p}")
    print("\nEvery tier is a prefix of the manifest, so the cache can be reused across tiers."
          "\nScale up by appending only -- reordering the prefix misaligns every {data_id}.pth.")


if __name__ == "__main__":
    main()
