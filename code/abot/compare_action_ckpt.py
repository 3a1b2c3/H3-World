#!/usr/bin/env python3
"""动作注入层的机制诊断：两个 checkpoint 一比，就知道注入通路活没活。

这是 `docs/action_injection_design.md` §6 的验证工具。**改完注入方式后先跑这个，
不要去渲视频** —— 8 分钟渲一条看"像不像"，远不如两个 checkpoint 差分来得快和客观。

三个判据（bias 版实测值列在括号里，作为失效基线）：

  1. 调制的相对量级       —— bias 版训到 2952 步只有 0.107%，且增速持续衰减
  2. ΔW 与 W 的方向余弦   —— bias 版仅 +0.064，几乎正交，说明梯度在互相抵消
  3. 实际步长 / Adam 理论 —— bias 版只有 1.92%，动量被反向梯度吃掉了

FiLM 版如果机制活了，2 和 3 应该明显高于这两个数。

用法:
    python3 compare_action_ckpt.py A.safetensors B.safetensors
    python3 compare_action_ckpt.py --dir output/minimax_h3_abot/7872     # 自动取最早/最新两个
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch
from safetensors.torch import load_file

# 注入点处 hidden 每行的 L2，由 video_patch_proj(真实 stage-1 latent) 实测得到。
# 换分辨率/换模型要重算：见 docs/action_injection_design.md §1.1。
HIDDEN_ROW_L2 = 224.55


def _step_of(path: str) -> int:
    m = re.search(r"step-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_action(path: str):
    """返回 (mode, {名字: [L, hidden, C] 张量})。自动识别 bias / film。"""
    sd = load_file(path)
    groups = {}
    for tag in ("action_scale", "action_shift", "action_embedder"):
        keys = sorted((k for k in sd if tag in k),
                      key=lambda k: int(re.search(r"\.(\d+)\.", k).group(1))
                      if re.search(r"\.(\d+)\.", k) else 0)
        if keys:
            groups[tag] = torch.stack([sd[k].float() for k in keys])
    if not groups:
        sys.exit(f"{path} 里没有任何动作注入层，键样例: {list(sd)[:4]}")
    mode = "film" if "action_scale" in groups else "bias"
    return mode, groups, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="*", help="两个 checkpoint；省略则配合 --dir 自动取")
    ap.add_argument("--dir", default=None, help="从该目录自动取最早和最新两个 step")
    ap.add_argument("--lr", type=float, default=1e-4, help="训练用的学习率（算 Adam 理论步长）")
    args = ap.parse_args()

    paths = list(args.ckpt)
    if args.dir:
        cands = sorted(glob.glob(os.path.join(args.dir, "step-*.safetensors")), key=_step_of)
        if len(cands) < 2:
            sys.exit(f"{args.dir} 里不足两个 checkpoint")
        paths = [cands[0], cands[-1]]
    if len(paths) != 2:
        sys.exit("需要恰好两个 checkpoint（或用 --dir）")

    a_mode, A, _ = load_action(paths[0])
    b_mode, B, sd_b = load_action(paths[1])
    if a_mode != b_mode:
        sys.exit(f"两个 checkpoint 的注入模式不同: {a_mode} vs {b_mode}")
    sa, sb = _step_of(paths[0]), _step_of(paths[1])
    dstep = max(sb - sa, 1)

    print(f"模式      : {a_mode}")
    print(f"对比      : step-{sa}  →  step-{sb}   （相差 {dstep} 步，lr={args.lr:g}）")
    n_lora = len([k for k in sd_b if "lora" in k.lower()])
    print(f"checkpoint: 动作层 {sum(v.shape[0] for v in B.values())} 个张量，LoRA {n_lora} 个\n")

    # ---- 判据 1：调制的相对量级 ----
    # film 看 scale（乘性项，直接就是相对调制比例）；bias 看列 L2 比 hidden 行 L2。
    key = "action_scale" if a_mode == "film" else "action_embedder"
    print("【判据 1】调制的相对量级")
    for tag, W in B.items():
        col = W.norm(dim=1)                     # [L, C] 每层每个通道的列 L2
        if tag == "action_scale":
            # scale 作用于 (1+scale)*h，per-element 均值就是调制比例
            rel = W.abs().mean().item()
            print(f"  {tag:16} 单层列L2 中位 {col.median():.4f}   "
                  f"平均 |scale| = {rel*100:.4f}%  ← 直接是调制比例")
        else:
            print(f"  {tag:16} 单层列L2 中位 {col.median():.4f}   "
                  f"相对 hidden({HIDDEN_ROW_L2}) = {col.median()/HIDDEN_ROW_L2*100:.4f}%")
    print(f"  基线（bias 版 2952 步）: 0.2396 / 0.107%\n")

    # ---- 判据 2 & 3：梯度是否在抵消 ----
    print("【判据 2/3】梯度方向一致性")
    for tag in B:
        Wa, Wb = A[tag], B[tag]
        d = Wb - Wa
        cos = torch.nn.functional.cosine_similarity(Wa.flatten(), d.flatten(), dim=0).item()
        per_step = d.abs().mean().item() / dstep
        ratio = per_step / args.lr
        aligned = Wb.norm().item() / (Wa.norm() + d.norm()).item()
        print(f"  {tag}")
        print(f"    ΔW 与 W 方向余弦      {cos:+.3f}      (bias 基线 +0.064)")
        print(f"    每步实际 / Adam 理论  {ratio*100:.2f}%      (bias 基线 1.92%)")
        print(f"    |W_new| / (|W|+|ΔW|)  {aligned:.3f}      "
              f"{'← 基本同向累积' if aligned > 0.9 else '← 有抵消'}")
    print()

    verdict = []
    for tag in B:
        d = B[tag] - A[tag]
        cos = torch.nn.functional.cosine_similarity(A[tag].flatten(), d.flatten(), dim=0).item()
        verdict.append(cos)
    best = max(verdict)
    if best > 0.4:
        print("结论: ✅ 方向一致性明显优于 bias 基线，注入通路在有效学习")
    elif best > 0.15:
        print("结论: ⚠️  比 bias 基线好，但仍有抵消 —— 再跑一段看趋势")
    else:
        print("结论: ❌ 与 bias 基线同级，机制没活，不要往下走")


if __name__ == "__main__":
    main()
