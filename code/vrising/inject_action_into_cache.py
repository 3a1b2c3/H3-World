#!/usr/bin/env python3
"""把动作张量事后注入到已有的 stage 1 latent 缓存里。

为什么可以事后注入：stage 1 存的 .pth 就是普通的
    tuple(inputs_shared: dict, inputs_posi: dict, inputs_nega: dict)
往 inputs_shared 里加一个 action_cond key 再存回即可，是纯磁盘 I/O。
form_wolf 的 15.7 GB 按 /nfs 实测 350 MB/s 大约 90 秒，比重跑 stage 1 便宜两个量级。

前提（两条都必须满足，否则 action 到不了 stage 2）：
  1. model_fn_minimax_h3 已有具名形参 action_cond —— stage 1 末尾的
     GeneralUnit_RemoveCache 按 inspect.signature(pipe.model_fn).parameters 组白名单，
     不在白名单的 key 会被剪掉。
  2. 缓存文件名 {data_id}.pth 里的 data_id 是 metadata 的行号，所以传进来的
     metadata 必须和跑 stage 1 时用的是同一个文件，行序不能变。

用法:
    python3 inject_action_into_cache.py --subset form_wolf
    python3 inject_action_into_cache.py --subset form_wolf --dry-run
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_action as A

DATA_BASE = "/nfs/danze/data/v_rising/vrising_train_bundle/data/transform_classified"
MODEL_BASE = "/nfs/danze/model/minimax_h3_vrising"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="form_wolf")
    ap.add_argument("--num-frames", type=int, default=107)
    ap.add_argument("--dry-run", action="store_true", help="只校验不写盘")
    ap.add_argument("--force", action="store_true", help="已有 action_cond 也重算覆盖")
    args = ap.parse_args()

    meta_path = os.path.join(DATA_BASE, f"h3_meta_{args.subset}.jsonl")
    cache_dir = os.path.join(MODEL_BASE, f"{args.subset}-cache")
    rows = [json.loads(l) for l in open(meta_path)]
    latent_t = A.latent_t_for(args.num_frames)
    print(f"metadata: {meta_path}  {len(rows)} 行")
    print(f"cache   : {cache_dir}")
    print(f"latent_t: {latent_t}  (num_frames={args.num_frames})\n")

    pth = {}
    for root, _, files in os.walk(cache_dir):
        for fn in files:
            if fn.endswith(".pth"):
                pth[int(fn[:-4])] = os.path.join(root, fn)
    print(f"发现缓存 {len(pth)} 个\n")

    n_ok = n_skip = n_miss = 0
    key_hist = torch.zeros(A.NUM_BUTTONS)
    for data_id in sorted(pth):
        if data_id >= len(rows):
            raise SystemExit(f"❌ data_id {data_id} 超出 metadata 行数 {len(rows)}——"
                             f"缓存和 metadata 对不上，行序可能变了")
        row = rows[data_id]
        side = A.sidecar_path_for(row["video"], DATA_BASE)
        if not os.path.exists(side):
            n_miss += 1
            continue

        act = A.action_from_sidecar(side, args.num_frames, row.get("category"))
        assert act.shape == (latent_t, A.NUM_BUTTONS), f"{side}: {act.shape}"
        key_hist += act.amax(0)

        path = pth[data_id]
        data = torch.load(path, map_location="cpu", weights_only=False)
        shared = data[0]

        # 与真实 latent 交叉校验时间轴：input_latents 是 (1, 24, latent_t, h, w)
        lt = shared["input_latents"].shape[2]
        if lt != latent_t:
            raise SystemExit(f"❌ {path}: 缓存 latent_t={lt} 与动作 latent_t={latent_t} 不符")

        if "action_cond" in shared and not args.force:
            n_skip += 1
            continue
        if args.dry_run:
            n_ok += 1
            continue

        shared["action_cond"] = act
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)          # 同目录 rename 原子，最终文件存在即完整
        n_ok += 1
        if n_ok % 200 == 0:
            print(f"  已注入 {n_ok}...")

    print(f"\n{'[dry-run] ' if args.dry_run else ''}注入 {n_ok}，跳过(已有) {n_skip}，缺 sidecar {n_miss}")
    print("按键覆盖（出现过该键的 clip 数）:")
    for k, c in zip(A.BUTTON_COLS, key_hist.tolist()):
        if c:
            print(f"   {k:12s}{int(c):5d}")


if __name__ == "__main__":
    main()
