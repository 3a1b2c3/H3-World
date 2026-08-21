#!/usr/bin/env python3
"""逐 latent 文本条件缓存的完整性校验。起训前必跑。

数据构建是事后重写 prompt_embeds + packed 的，出错不会当场报警 —— 只会在训练里
静默地表现为"学不到绑定"。所以每条都要逐项核，而不是抽查。

判据（任一条不过就不该起训）：
  1. action_cond 已删除，action_text_rows / video_start / frame_rows 齐全
  2. prompt_embeds 行数 == text_len == packed 里的 text_pos 长度
  3. 37 条标注行区间：升序、无缝、不越界、不与头部重叠
  4. 镜像偏移：每条标注与自己那帧的 t 偏移是同一个负常数
  5. token_tags：头部保留原标签（图像 pad 走 video 组），标注行全为文本
  6. 视频/音频 latent 未被动过（形状仍是 stage 1 的）

用法:
    python3 code/abot/verify_text_cache.py --cache output/minimax_h3_abot/7872-cache
    python3 code/abot/verify_text_cache.py --cache ... --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

LATENT_T = 37


def check_one(path: str) -> tuple[list[str], dict]:
    bad = []
    kwargs, extra, *_ = torch.load(path, map_location="cpu", weights_only=False)
    packed, pe = extra["packed"], extra["prompt_embeds"]

    if "action_cond" in kwargs:
        bad.append("action_cond 未删除")
    for k in ("action_text_rows", "action_video_start", "action_frame_rows"):
        if k not in packed:
            bad.append(f"缺 {k}")
    if bad:
        return bad, {}

    rows = packed["action_text_rows"]
    v0, fr = int(packed["action_video_start"]), int(packed["action_frame_rows"])
    text_len = int(packed["text_pos"].numel())
    head_len = int(rows[0, 0])

    if int(pe.shape[0]) != text_len:
        bad.append(f"prompt_embeds {pe.shape[0]} != text_len {text_len}")
    if tuple(rows.shape) != (LATENT_T, 2):
        bad.append(f"action_text_rows 形状 {tuple(rows.shape)}")
    else:
        if int(rows[-1, 1]) != text_len:
            bad.append(f"末条标注止于 {int(rows[-1,1])} != text_len {text_len}")
        for k in range(LATENT_T):
            lo, hi = int(rows[k, 0]), int(rows[k, 1])
            if not (0 <= lo < hi <= text_len):
                bad.append(f"第 {k} 条区间 ({lo},{hi}) 越界"); break
            if k and lo != int(rows[k - 1, 1]):
                bad.append(f"第 {k} 条与前一条不衔接"); break

        g = packed["img_position_ids"][0]
        t_ann = g[rows[:, 0], 0]
        t_frm = g[torch.tensor([v0 + k * fr for k in range(LATENT_T)]), 0]
        d = (t_ann - t_frm)
        if float(d.max() - d.min()) > 1e-6:
            bad.append(f"镜像偏移不恒定: {float(d.min()):.2f}..{float(d.max()):.2f}")
        elif float(d[0]) >= 0:
            bad.append(f"偏移非负 {float(d[0]):.1f}（文本应在视频之前）")
        if bool((t_ann >= text_len).any()):
            bad.append("有标注的 t 越过 text_len")

    tags = packed["token_tags"][packed["text_pos"]]
    if int((tags[head_len:] != 1).sum()):
        bad.append("标注行的 token_tags 不全为 1")
    if int((tags[:head_len] == 0).sum()) == 0:
        bad.append("头部没有视觉行")

    il = kwargs["input_latents"].shape
    if len(il) != 5 or il[2] != LATENT_T:
        bad.append(f"input_latents 形状异常 {tuple(il)}")

    return bad, {"text_len": text_len, "head_len": head_len,
                 "seq_len": int(packed["seq_len"]), "offset": float(d[0]) if not bad else 0.0,
                 "n_img": int((tags[:head_len] == 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default=None, help="把统计写成 json（分片汇总用）")
    args = ap.parse_args()

    paths = []
    for root, _, files in os.walk(args.cache):
        paths += [os.path.join(root, f) for f in files if f.endswith(".pth")]
    paths.sort(key=lambda p: int(os.path.basename(p)[:-4]))
    mine = paths[args.shard::args.num_shards]
    tag = f"[{args.shard}/{args.num_shards}] " if args.num_shards > 1 else ""
    print(f"{tag}校验 {len(mine)} / {len(paths)} 条")

    stats, failures = [], []
    for n, p in enumerate(mine):
        bad, st = check_one(p)
        if bad:
            failures.append((os.path.basename(p), bad))
        else:
            stats.append(st)
        if (n + 1) % 200 == 0 or n + 1 == len(mine):
            print(f"{tag}  {n + 1}/{len(mine)}  失败 {len(failures)}", flush=True)

    summary = {"n": len(mine), "ok": len(stats), "fail": len(failures),
               "failures": failures[:20]}
    if stats:
        for key in ("text_len", "head_len", "seq_len", "n_img"):
            v = np.array([s[key] for s in stats])
            summary[key] = [int(v.min()), int(v.max()), float(v.mean())]
        summary["offsets"] = sorted({round(s["offset"], 3) for s in stats})
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False))
    print(f"{tag}通过 {len(stats)} / {len(mine)}")
    for name, bad in failures[:10]:
        print(f"{tag}  ✗ {name}: {'; '.join(bad)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
