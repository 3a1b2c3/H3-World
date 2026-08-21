#!/usr/bin/env python3
"""把每卡一条样本的 N 个 run 目录合成一个，供 build_compare_page.py 用。

infer_abot.py 一个进程写一个 run 目录；8 卡并行就有 8 份 manifest。
合并时按给定顺序拼 samples，config/checkpoint 取第一份并核对一致
（checkpoint 不同就是拿混了，直接报错而不是悄悄拼出一张假对比页）。
"""
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path("/opt/dlami/nvme/danze/minimax_finetune/output/abot_inference")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="run 目录名前缀，如 step3000_film_gpu")
    ap.add_argument("--out", required=True, help="合并后的目录名")
    args = ap.parse_args()

    dirs = sorted(ROOT.glob(args.pattern + "*"), key=lambda d: int(d.name.rsplit("gpu", 1)[-1]))
    dirs = [d for d in dirs if (d / "manifest.json").exists()]
    if not dirs:
        sys.exit(f"没找到 {args.pattern}* 下的 manifest.json")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    base, samples = None, []
    for d in dirs:
        man = json.loads((d / "manifest.json").read_text())
        if base is None:
            base = man
        elif man["checkpoint"]["path"] != base["checkpoint"]["path"]:
            sys.exit(f"{d.name} 的 checkpoint 与 {dirs[0].name} 不一致，拒绝合并")
        for s in man["samples"]:
            if s.get("status") not in (None, "ok", "complete"):
                print(f"跳过 {s['sample_id']}（status={s['status']}）"); continue
            src = d / f"{s['sample_id']}_w{s['window']:03d}"
            if not src.is_dir():
                print(f"跳过 {s['sample_id']}（目录缺失）"); continue
            dst = out / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            samples.append(s)

    base["samples"] = samples
    base["merged_from"] = [d.name for d in dirs]
    (out / "manifest.json").write_text(json.dumps(base, ensure_ascii=False, indent=2))
    print(f"合并 {len(samples)} 条样本 -> {out}")

if __name__ == "__main__":
    main()
