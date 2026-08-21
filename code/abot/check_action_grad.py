#!/usr/bin/env python3
"""读 stage 2 存盘，逐通道检查 action_embedders 学到了什么。

这是 `docs/BUNDLE_README.md` §6.3 的两条判据的实现。为什么不看 loss 曲线：
flow matching 每步随机采 timestep，方差压倒性地大，5000 步只从 0.190 挪到 0.180，
没有判据性。存盘权重的逐通道范数才有。

判据一（通路正确性）：
    梯度非零的通道集合必须**精确等于**数据里出现过的通道。
    Q/E/Space 在 ABot 数据里恒零，对应列必须**精确**为 0 ——
    零初始化下 dW[:,j] = Σ_k dL/demb[k] · action[k,j]，该列输入恒零则梯度精确为 0。
    这一条是数学必然，如果不成立，说明通道对错位了。

判据二（捷径学习）：
    比按键通道与相机通道的 |w|max。按键明显偏低 ⇒ 模型主要在读相机、
    按键通路没被训起来。注意这**不是**梯度量级不足造成的：实测每条 clip 流经
    按键 11 通道的信号总量与相机 6 通道相当（相机/按键 ≈ 1.09x），
    所以按键真学不动的话，那是模型主动忽略，不是没有梯度。
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abot_action as A

ROOT = os.environ.get("ABOT_OUT_ROOT", "/opt/dlami/nvme/danze/minimax_finetune/data")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None, help="存盘 .safetensors；默认取目录里最新的")
ap.add_argument("--dir", default=f"{ROOT}/../output/minimax_h3_abot/64")
args = ap.parse_args()

path = args.ckpt
if path is None:
    cands = sorted(glob.glob(os.path.join(args.dir, "**", "*.safetensors"), recursive=True),
                   key=os.path.getmtime)
    if not cands:
        sys.exit(f"没找到存盘: {args.dir}")
    path = cands[-1]
print(f"存盘: {path}\n")

from safetensors.torch import load_file
sd = load_file(path)
keys = [k for k in sd if "action_embedder" in k]
if not keys:
    sys.exit(f"存盘里没有 action_embedders，键样例: {list(sd)[:5]}")
print(f"action_embedders 层数: {len(keys)}   其余键: {len(sd) - len(keys)}\n")

# 每层是 Linear(num_buttons -> hidden) 的 weight，形状 [hidden, num_buttons]
W = torch.stack([sd[k].float() for k in sorted(keys)])        # [L, hidden, C]
col_absmax = W.abs().amax(dim=(0, 1))                          # 每个通道跨层的 |w|max
col_l2 = W.pow(2).sum(dim=(0, 1)).sqrt()

channel_sets = {
    len(A.ACTIVE_KEY_COLS): A.ACTIVE_KEY_COLS,
    len(A.KEY_COLS): A.KEY_COLS,
    len(A.ACTION_COLS): A.ACTION_COLS,
}
channel_names = channel_sets.get(W.shape[-1])
if channel_names is None:
    sys.exit(f"不支持的动作维度 {W.shape[-1]}，预期 8、11 或 17")

print(f"{'通道':<12}{'|w|max':>12}{'L2':>12}   类型   判定")
dead = []
for j, name in enumerate(channel_names):
    kind = "按键" if name in A.KEY_COLS else "相机"
    exact_zero = col_absmax[j].item() == 0.0
    if name in ("Q", "E", "Space"):
        ok = "✅ 精确为 0（数据恒零，符合预期）" if exact_zero else "❌ 非零 —— 通道错位！"
    else:
        ok = "❌ 精确为 0 —— 该通道没拿到梯度" if exact_zero else "有梯度"
        if exact_zero:
            dead.append(name)
    print(f"{name:<12}{col_absmax[j].item():12.3e}{col_l2[j].item():12.3e}   {kind}   {ok}")

kb_indices = [j for j, name in enumerate(channel_names) if name in A.ACTIVE_KEY_COLS]
live_kb = col_absmax[kb_indices]
dead_channels_present = all(name in channel_names for name in ("Q", "E", "Space"))
if dead_channels_present:
    print(f"\n判据一：Q/E/Space 精确为 0 —— "
          f"{'✅ 成立' if all(col_absmax[channel_names.index(n)].item() == 0.0 for n in ('Q','E','Space')) else '❌ 不成立'}")
else:
    print("\n判据一：8 维配置已移除 Q/E/Space；检查所有保留通道均有梯度")
if dead:
    print(f"        ⚠️ 另有本应有信号的通道为零: {dead}")

print(f"\n保留按键 |w|max 中位 {live_kb.median():.3e}")
cam_indices = [j for j, name in enumerate(channel_names) if name not in A.KEY_COLS]
if cam_indices:
    cam = col_absmax[cam_indices]
    print(f"相机通道 |w|max 中位 {cam.median():.3e}")
    r = (live_kb.median() / cam.median()).item()
    print(f"按键/相机 = {r:.2f}x   ", end="")
    if r < 0.3:
        print("⚠️ 按键明显偏低 —— 模型主要在读相机，捷径学习发生了")
    elif r < 0.7:
        print("按键偏低，需要关注")
    else:
        print("✅ 两类通道量级相当，按键通路确实在训")
elif not dead:
    print("✅ 8 个动作通道都获得了梯度")
