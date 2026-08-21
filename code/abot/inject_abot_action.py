#!/usr/bin/env python3
"""把 ABot 的动作张量事后注入 stage 1 的 latent 缓存。

沿用 V-Rising 那套（`/nfs/danze/inject_action_into_cache.py`）的思路与原子写法，
两处不同：

  * 动作来自 `build_abot_clips.py` 预先算好的 `<clip>.npy`（[130, 17]，
    逐帧、未缩放），这里只做「截到 num_frames + 分箱 + 缩放」。
    好处是**动作 schema 不被 stage 1 锁死** —— 想改通道构成或量纲，
    重跑本脚本几分钟就行，不用重跑 stage 1 的 latent 编码。
  * 支持 `--cam-dropout`：按 clip 确定性地把相机增量通道整块置零。
    这是防「捷径学习」的：相机增量是动作的**结果**，信息量远大于按键，
    模型可能干脆只读它、让按键通路彻底不训练。留一部分样本只有按键，
    强迫按键通路也得干活。用 clip 下标播种，跨 epoch 固定，
    等价于把数据集分成「带相机条件」和「纯按键」两个子集。

前提（与 V-Rising 完全相同，破一条 action 就到不了 stage 2）：
  1. `model_fn_minimax_h3` 有具名形参 `action_cond`（缓存白名单来自
     `inspect.signature`，`**kwargs` 不算数）。
  2. `{data_id}.pth` 的 data_id 就是 metadata 行号，metadata 必须与跑
     stage 1 时逐行相同。

用法:
    python3 inject_abot_action.py --meta abot_meta_64.jsonl --cache <...>-cache --dry-run
    python3 inject_abot_action.py --meta abot_meta_64.jsonl --cache <...>-cache
    python3 inject_abot_action.py ... --active-keys-only # 只留 8 个有效按键通道
    python3 inject_abot_action.py ... --keys-only        # 保留全部 11 个按键通道
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abot_action as A

OUT_ROOT = os.environ.get("ABOT_OUT_ROOT", "/opt/dlami/nvme/danze/minimax_finetune/data")
CLIP_DIR = f"{OUT_ROOT}/clips"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="metadata jsonl（文件名或绝对路径）")
    ap.add_argument("--cache", required=True, help="stage 1 的 -cache 目录")
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--rot-scale", type=float, default=A.ROT_SCALE)
    ap.add_argument("--tra-scale", type=float, default=A.TRA_SCALE)
    ap.add_argument("--tra-clip", type=float, default=A.TRA_CLIP)
    ap.add_argument("--rot-clip", type=float, default=A.ROT_CLIP)
    ap.add_argument("--cam-dropout", type=float, default=0.5,
                    help="按 clip 确定性置零相机增量通道的比例；0 表示全保留")
    ap.add_argument("--keys-only", action="store_true", help="丢掉 6 个相机通道，只留 11 键")
    ap.add_argument("--active-keys-only", action="store_true",
                    help="只留有信号的 W/A/S/D/I/J/K/L 8 个按键通道")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="已有 action_cond 也覆盖")
    ap.add_argument("--require-complete", action="store_true",
                    help="要求缓存数量和 metadata 行数完全一致")
    ap.add_argument("--expected-count", type=int, default=None,
                    help="要求恰好有 N 个缓存（用于 metadata 前 N 条 smoke）")
    args = ap.parse_args()

    if args.keys_only and args.active_keys_only:
        ap.error("--keys-only 和 --active-keys-only 不能同时使用")

    meta = args.meta if (os.path.isabs(args.meta) or os.path.isfile(args.meta)) else os.path.join(OUT_ROOT, args.meta)
    rows = [json.loads(l) for l in open(meta)]
    latent_t = A.latent_t_for(args.num_frames)
    if args.active_keys_only:
        action_cols = A.ACTIVE_KEY_COLS
    elif args.keys_only:
        action_cols = A.KEY_COLS
    else:
        action_cols = A.ACTION_COLS
    dim = len(action_cols)

    print(f"metadata : {meta}  {len(rows)} 行")
    print(f"cache    : {args.cache}")
    print(f"latent_t : {latent_t}  (num_frames={args.num_frames})")
    print(f"action   : dim={dim}  rot/{args.rot_scale}  tra/{args.tra_scale}  "
          f"columns={','.join(action_cols)}  "
          f"cam_dropout={0.0 if (args.keys_only or args.active_keys_only) else args.cam_dropout}")
    print(f"→ stage 2 必须带 --action_num_buttons {dim}\n")

    pth = {}
    for root, _, files in os.walk(args.cache):
        for fn in files:
            if fn.endswith(".pth"):
                data_id = int(fn[:-4])
                if data_id in pth:
                    raise SystemExit(f"❌ 重复 data_id {data_id}: {pth[data_id]} 和 {os.path.join(root, fn)}")
                pth[data_id] = os.path.join(root, fn)
    print(f"发现缓存 {len(pth)} 个\n")
    if args.expected_count is not None and len(pth) != args.expected_count:
        raise SystemExit(f"❌ 缓存 {len(pth)} 个，预期 {args.expected_count} 个")
    if args.require_complete and len(pth) != len(rows):
        raise SystemExit(f"❌ 缓存 {len(pth)} 个，metadata {len(rows)} 行，不完整")

    n_ok = n_skip = n_drop = 0
    key_hist = np.zeros(dim if args.active_keys_only else A.NUM_KEYS)
    cam_absmax = np.zeros(6)
    for data_id in sorted(pth):
        if data_id >= len(rows):
            raise SystemExit(f"❌ data_id {data_id} 超出 metadata 行数 {len(rows)}"
                             f" —— 缓存与 metadata 行序对不上")
        row = rows[data_id]
        mat = np.load(os.path.join(CLIP_DIR, row["action"]))
        if mat.shape[0] < args.num_frames:
            raise SystemExit(f"❌ {row['action']}: 只有 {mat.shape[0]} 帧，"
                             f"不足 num_frames={args.num_frames}")
        act = A.bin_to_latent(mat[:args.num_frames], latent_t,
                              rot_scale=args.rot_scale, tra_scale=args.tra_scale,
                              tra_clip=args.tra_clip, rot_clip=args.rot_clip)

        if args.active_keys_only:
            act = act[:, A.ACTIVE_KEY_INDICES]
        elif args.keys_only:
            act = act[:, :A.NUM_KEYS]
        elif args.cam_dropout > 0:
            # 用 sample_id 播种而不是 data_id：换 tier 时同一条 clip 的归属不变
            h = int.from_bytes(row["sample_id"][:8].encode(), "big")
            if (h % 1000) / 1000.0 < args.cam_dropout:
                act[:, A.NUM_KEYS:] = 0.0
                n_drop += 1

        if act.shape != (latent_t, dim):
            raise SystemExit(f"❌ {row['action']}: 分箱后 {act.shape} != {(latent_t, dim)}")
        key_hist += (act[:, :len(key_hist)].max(axis=0) > 0)
        if not (args.keys_only or args.active_keys_only):
            cam_absmax = np.maximum(cam_absmax, np.abs(act[:, A.NUM_KEYS:]).max(axis=0))

        path = pth[data_id]
        data = torch.load(path, map_location="cpu", weights_only=False)
        shared = data[0]
        # 拿真实 latent 交叉校验时间轴：input_latents 是 (1, 24, latent_t, h, w)
        lt = shared["input_latents"].shape[2]
        if lt != latent_t:
            raise SystemExit(f"❌ {path}: 缓存 latent_t={lt} 与动作 latent_t={latent_t} 不符")

        if "action_cond" in shared and not args.force:
            if tuple(shared["action_cond"].shape) != (latent_t, dim):
                raise SystemExit(
                    f"❌ {path}: 已有 action_cond {tuple(shared['action_cond'].shape)}，"
                    f"预期 {(latent_t, dim)}；确认配置后使用 --force 覆盖"
                )
            n_skip += 1
            continue
        if args.dry_run:
            n_ok += 1
            continue

        shared["action_cond"] = torch.from_numpy(act)
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)              # 同目录 rename 原子
        n_ok += 1
        if n_ok % 200 == 0:
            print(f"  已注入 {n_ok}...")

    print(f"\n{'[dry-run] ' if args.dry_run else ''}注入 {n_ok}，跳过(已有) {n_skip}，"
          f"相机通道被置零 {n_drop}")
    print("\n按键覆盖（出现过该键的 clip 数）:")
    printed_key_cols = A.ACTIVE_KEY_COLS if args.active_keys_only else A.KEY_COLS
    for k, c in zip(printed_key_cols, key_hist):
        flag = "  ← 恒零，该通道拿不到梯度" if c == 0 else ""
        print(f"   {k:8s}{int(c):6d}{flag}")
    if not (args.keys_only or args.active_keys_only):
        print("\n相机通道 |max|（分箱缩放后，理想落在 O(1)）:")
        for k, v in zip(A.ROT_COLS + A.TRA_COLS, cam_absmax):
            print(f"   {k:12s}{v:8.3f}")


if __name__ == "__main__":
    main()
