#!/usr/bin/env python3
"""Write the per-latent action-text condition into the stage-1 latent cache
after the fact.

Same idea as an additive-conditioning injector (post-hoc, atomic write,
data_id = metadata row number), except this one rewrites
**prompt_embeds + packed** instead of adding an action_cond tensor:

  * the video/audio latents are untouched -- changing the annotation scheme
    never requires re-running stage 1's VAE encoding
  * the text segment becomes [original head (image pad + scene prompt) |
    37 annotation rows], with each annotation placed at a mirrored offset
  * `packed` carries out action_text_rows / action_video_start /
    action_frame_rows, which the DiT uses to build the hard binding mask
    (annotation row k is only mutually visible with frame k's video rows)
  * **action_cond is dropped** -- this scheme no longer uses the
    FiLM/bias-style feature-space injection

**The head (first-frame vision + scene prompt) is re-encoded every time**,
through `presentation_fl2va(prompt, image_counts)` -- the exact same call
used at inference, so training and inference are consistent by
construction and don't depend on whatever head happened to be in the cache
already (an earlier version reused the cached head, which produced two
different head structures once the cache had been rewritten once).
Measured at 0.25 s/clip.

The 37 annotations are **encoded independently, one at a time**: the same
string always produces the same embedding. This is both what makes the
dedup dictionary valid and a hard requirement for streaming inference
(future actions are unknown at inference time, so the whole sequence can't
be encoded together).

Idempotent: the head is re-encoded from source data every run and the
annotation segment is rebuilt from scratch, independent of the cache's
current state, so repeated runs give the same result and a failed run can
just be rerun.

Usage:
    python3 code/abot/inject_abot_text.py --meta data/abot_meta_train_7872.jsonl \\
        --cache output/minimax_h3_abot/7872-cache --dry-run
    python3 code/abot/inject_abot_text.py --meta ... --cache ... --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSYNTH_ROOT = Path(os.environ.get("ABOT_DIFFSYNTH_ROOT", str(PROJECT_ROOT / "DiffSynth-Studio-h3-v2")))
CACHE_ROOT = PROJECT_ROOT / ".cache"
for _n, _p in {
    "HF_HOME": CACHE_ROOT / "hf", "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "TORCH_HOME": CACHE_ROOT / "torch", "TMPDIR": CACHE_ROOT / "tmp",
}.items():
    os.environ.setdefault(_n, str(_p))
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(DIFFSYNTH_ROOT / "models")
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(DIFFSYNTH_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import abot_action as A  # noqa: E402
import action_script as S  # noqa: E402

LATENT_T = 37
NUM_FRAMES = 124
AUDIO_ROWS = 414              # audio_latent_t(207) * 2 channels; fixed by num_frames, independent of resolution

# The three resolution-dependent quantities are all derived from
# (HEIGHT, WIDTH) rather than hardcoded:
#   frame_rows      rows per latent frame = (H/16/2) * (W/16/2)
#   IMAGE_PAD_ROWS  rows the first frame takes through the vision tower = frame_rows + 2
#                   (one vision special token before and after)
# 480x832 -> 390 / 392, 768x1344 -> 1008 / 1010, both verified against the measured image_token_counts.
HEIGHT = int(os.environ.get("ABOT_HEIGHT", 480))
WIDTH = int(os.environ.get("ABOT_WIDTH", 832))


def frame_rows_of(height: int, width: int) -> int:
    if height % 32 or width % 32:
        raise SystemExit(f"height and width must both be multiples of 32 (VAE 16x * patch 2x2), got {width}x{height}")
    return (height // 32) * (width // 32)


FRAME_ROWS = frame_rows_of(HEIGHT, WIDTH)
IMAGE_PAD_ROWS = FRAME_ROWS + 2


def collect_scripts(rows, clip_root: Path) -> tuple[list[list[str]], dict[str, None]]:
    """The 37 annotation sentences for every sample, plus the deduplicated vocabulary."""
    scripts, vocab = [], {}
    for r in rows:
        pooled = A.bin_to_latent(np.load(clip_root / r["action"])[:NUM_FRAMES], LATENT_T)
        sc = S.annotate_from_keys9(S.keys9(pooled))
        scripts.append(sc)
        for line in sc:
            vocab.setdefault(line, None)
    return scripts, vocab


def scan_pad_used_to(rows, clip_root: Path, tokenizer, align: int = 64) -> tuple[int, dict]:
    """Scan the whole dataset once to compute a single shared padding length.

    This has to be uniform: flex_attention is torch.compile'd, and every
    distinct sequence length triggers a recompile; with a different
    text_len per sample, this would blow past Dynamo's recompile limit
    within a few steps.

    This scan is cheap -- head_len can be computed exactly from the
    tokenizer alone, without running the vision tower or encoder:
    presentation_fl2va = prefix + image pad (constant 392) + prompt, and
    all three segment lengths can be counted directly.
    """
    from diffsynth.models.minimax_h3_text_encoder import presentation_fl2va

    prefix = len(presentation_fl2va(tokenizer, "", [IMAGE_PAD_ROWS])[0]) - IMAGE_PAD_ROWS
    ann_len: dict[str, int] = {}
    stats = {"used": [], "text_len": []}
    for r in rows:
        pooled = A.bin_to_latent(np.load(clip_root / r["action"])[:NUM_FRAMES], LATENT_T)
        script = S.annotate_from_keys9(S.keys9(pooled))
        for line in script:
            if line not in ann_len:
                ann_len[line] = len(tokenizer(line)["input_ids"])
        head = prefix + IMAGE_PAD_ROWS + len(tokenizer(r["prompt"])["input_ids"])
        text_len = head + sum(ann_len[x] for x in script)
        # the cond / audio / video row counts are fixed by the latent shape, constant across this dataset
        used = text_len + FRAME_ROWS + AUDIO_ROWS + LATENT_T * FRAME_ROWS
        stats["used"].append(used)
        stats["text_len"].append(text_len)
    mx = max(stats["used"])
    return ((mx + align - 1) // align) * align, stats


def load_encoder(device: str, dtype=torch.bfloat16):
    """Load only the text_encoder + processor, leaving the DiT and VAE untouched."""
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    cfg = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype,
               onload_device=device, preparing_dtype=dtype, preparing_device=device,
               computation_dtype=dtype, computation_device=device)
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=[ModelConfig(model_id="MiniMax/MiniMax-H3",
                                   origin_file_pattern="FL2VA/text_encoder/model*.safetensors",
                                   **cfg)],
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3",
                                     origin_file_pattern="FL2VA/processor/"),
    )
    pipe.load_models_to_device(("text_encoder",))
    return pipe


def encode_vocab(pipe, vocab, device: str, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    """Encode each annotation **independently**: the same string always produces the same embedding."""
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va
    out = {}
    for i, line in enumerate(vocab):
        ids, _tags = presentation_t2va(pipe.tokenizer, line)
        ids = ids.unsqueeze(0).to(device)
        with torch.no_grad():
            h = pipe.text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids))
        out[line] = h.to("cpu", dtype)
        if (i + 1) % 20 == 0 or i + 1 == len(vocab):
            print(f"    encoded {i + 1}/{len(vocab)}", flush=True)
    return out


def encode_head(pipe, row, clip_root: Path, device: str, height=None, width=None):
    """First-frame vision + scene prompt, through the same presentation_fl2va call used at inference."""
    import imageio.v2 as iio
    from PIL import Image
    from diffsynth.models.minimax_h3_text_encoder import presentation_fl2va
    from diffsynth.pipelines.minimax_h3_audio_video import image_token_counts

    height = HEIGHT if height is None else height
    width = WIDTH if width is None else width
    reader = iio.get_reader(str(clip_root / row["video"]), "ffmpeg")
    frame = Image.fromarray(reader.get_data(0)); reader.close()
    frame = frame.convert("RGB").resize((width, height), Image.LANCZOS)
    pv, grid, counts = image_token_counts(pipe.processor, [frame])
    ids, tags = presentation_fl2va(pipe.tokenizer, row["prompt"], counts)
    ids = ids.unsqueeze(0).to(device)
    with torch.no_grad():
        h = pipe.text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                              pixel_values=pv.to(device, pipe.torch_dtype),
                              image_grid_thw=grid.to(device, torch.long))
    return h.to("cpu", pipe.torch_dtype), tags.view(-1).to(torch.long)


def rewrite_one(path: str, script: list[str], head, emb: dict[str, torch.Tensor],
                builder, dry: bool, pad_used_to: int | None = None) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    kwargs, extra = data[0], data[1]
    packed, old_pe = extra["packed"], extra["prompt_embeds"]
    head_rows, head_tags = head
    head_len = int(head_rows.shape[0])
    old_tl = int(packed["text_pos"].numel())

    parts = [head_rows.to(old_pe.dtype)]
    cursor = head_len
    spans = []
    for line in script:
        e = emb[line].to(old_pe.dtype)
        spans.append((cursor, cursor + int(e.shape[0])))
        cursor += int(e.shape[0])
        parts.append(e)
    new_pe = torch.cat(parts, dim=0)
    text_len = int(new_pe.shape[0])

    _, _, lt, lh, lw = kwargs["input_latents"].shape
    audio_t = int(kwargs["audio_input_latents"].shape[-1])
    frame_rows = (lh // 2) * (lw // 2)
    n_key = int(kwargs["keyframe_cond_anchor"].shape[0]) // frame_rows
    keyframe_indices = [0] if n_key == 1 else [0, -1]

    new_packed = builder._build_packed_fl2va(
        text_len, lt, lh, lw, audio_t, keyframe_indices,
        action_text_spans=spans, pad_used_to=pad_used_to)
    tt = new_packed["token_tags"]
    tt[new_packed["text_pos"]] = 1                         # every annotation row is text
    tt[new_packed["text_pos"][:head_len]] = head_tags      # the head keeps its original tags (image pad goes in the video group)

    stat = {"text_len": text_len, "seq_len": int(new_packed["seq_len"]),
            "head_len": head_len, "n_img": int((head_tags == 0).sum()),
            "real_used": int(new_packed["action_real_used"]),
            "old_text_len": old_tl, "old_seq_len": int(packed["seq_len"])}
    if dry:
        return stat

    extra["prompt_embeds"] = new_pe
    extra["packed"] = new_packed
    kwargs.pop("action_cond", None)                        # this scheme no longer uses feature-space injection
    tmp = path + f".tmp.{os.getpid()}"
    torch.save(data, tmp)
    os.replace(tmp, path)
    return stat


def main() -> None:
    global HEIGHT, WIDTH, FRAME_ROWS, IMAGE_PAD_ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--clip-root", default=str(PROJECT_ROOT / "data/clips"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0, help="which shard, 0-indexed")
    ap.add_argument("--num-shards", type=int, default=1, help="total shard count; use 8 for 8-GPU parallel")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N rows, for verification")
    ap.add_argument("--pad-used-to", type=int, default=None,
                    help="shared padding length; if omitted, scans the whole dataset to compute it (recommended)")
    ap.add_argument("--dry-run", action="store_true", help="compute only, don't write")
    ap.add_argument("--height", type=int, default=HEIGHT, help="training height, must be a multiple of 32")
    ap.add_argument("--width", type=int, default=WIDTH, help="training width, must be a multiple of 32")
    args = ap.parse_args()

    HEIGHT, WIDTH = args.height, args.width
    FRAME_ROWS = frame_rows_of(HEIGHT, WIDTH)
    IMAGE_PAD_ROWS = FRAME_ROWS + 2
    print(f"resolution {WIDTH}x{HEIGHT} -> frame_rows {FRAME_ROWS}, image pad {IMAGE_PAD_ROWS} rows")

    meta = Path(args.meta)
    if not meta.is_absolute():
        meta = PROJECT_ROOT / meta
    rows = [json.loads(l) for l in open(meta)]
    clip_root = Path(args.clip_root)

    pth = {}
    for root, _, files in os.walk(args.cache):
        for fn in files:
            if fn.endswith(".pth"):
                did = int(fn[:-4])
                if did in pth:
                    sys.exit(f"duplicate data_id {did}")
                pth[did] = os.path.join(root, fn)
    ids = sorted(pth)
    if args.limit:
        ids = ids[: args.limit]
    for did in ids:
        if did >= len(rows):
            sys.exit(f"data_id {did} exceeds the {len(rows)} metadata rows")
    ids = ids[args.shard::args.num_shards]
    tag = f"[{args.shard}/{args.num_shards}] " if args.num_shards > 1 else ""
    print(f"{tag}metadata has {len(rows)} rows, cache has {len(pth)} entries, this shard: {len(ids)}")

    scripts, vocab = collect_scripts([rows[i] for i in ids], clip_root)
    print(f"{tag}annotation vocabulary: {len(vocab)} distinct sentences")

    pipe = load_encoder(args.device)
    pad_used_to = args.pad_used_to
    if pad_used_to is None:
        # Important: scan **all** ids, not just this shard, or different shards would compute different values.
        all_ids = sorted(pth)[: args.limit] if args.limit else sorted(pth)
        pad_used_to, sc = scan_pad_used_to([rows[i] for i in all_ids], clip_root, pipe.tokenizer)
        print(f"{tag}shared padding: used {min(sc['used'])}-{max(sc['used'])} -> {pad_used_to}")
    emb = encode_vocab(pipe, vocab, args.device)
    lens = [int(v.shape[0]) for v in emb.values()]
    print(f"{tag}per-annotation length {min(lens)}-{max(lens)} rows (mean {sum(lens) / len(lens):.1f})")

    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Unit_PackedSequenceBuilder as U)
    builder = U.__new__(U)

    stats = []
    for n, (did, sc) in enumerate(zip(ids, scripts)):
        head = encode_head(pipe, rows[did], clip_root, args.device)
        stats.append(rewrite_one(pth[did], sc, head, emb, builder, args.dry_run,
                                 pad_used_to=pad_used_to))
        if (n + 1) % 100 == 0 or n + 1 == len(ids):
            print(f"{tag}  {'dry-run' if args.dry_run else 'written'} {n + 1}/{len(ids)}", flush=True)

    tl = np.array([s["text_len"] for s in stats])
    sl = np.array([s["seq_len"] for s in stats])
    hl = np.array([s["head_len"] for s in stats])
    osl = np.array([s["old_seq_len"] for s in stats])
    print(f"\n{tag}head_len {hl.min()}-{hl.max()} (re-encoded, of which {stats[0]['n_img']} rows are image pad)")
    print(f"{tag}text_len {tl.min()}-{tl.max()} (mean {tl.mean():.0f})")
    ru = np.array([s["real_used"] for s in stats])
    print(f"{tag}real_used {ru.min()}-{ru.max()}  ->  padded to a shared {pad_used_to}")
    print(f"{tag}seq_len  {osl.min()}-{osl.max()} -> {sl.min()}-{sl.max()}"
          f" (distinct values: {sorted(set(sl.tolist()))})")
    print(f"{tag}" + ("dry run complete, nothing written" if args.dry_run else "write complete, action_cond removed"))


if __name__ == "__main__":
    main()
