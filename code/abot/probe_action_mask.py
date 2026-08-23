#!/usr/bin/env python3
"""逐 latent 文本条件的硬绑定掩码 —— 落点探针。

掩码是这套方案的承重墙：自注意力对 token 置换等变，光把 37 条标注塞进序列，
模型无从知道哪条对哪帧。所以它错了不会报错，只会安静地训出一个学不到绑定的模型。
上线前必须逐条验死。

四条判据：
  1. 不传 action_text_rows 时，与改造前的注意力**逐位相同**（去掉标注即回到原模型）
  2. 标注行 k 只看得见第 k 帧的 video 行，看不见第 j≠k 帧
  3. video×video、锚点×video、标注×标注、标注×cond 全部未被触碰
  4. 与显式布尔掩码的参考实现逐位一致
  5. padding 行被屏蔽：真实行看不见 pad 行，pad 行只看得见自己（否则 softmax 出 NaN）
  6. **掩码复用一致性**：连续用多个不同的掩码，每个都必须对。这条最要紧 ——
     flex_attention 是 torch.compile 过的，mask_mod 闭包会被内联进 kernel；
     早先每条样本新建闭包时，实测第二次调用会复用第一次烘焙进去的掩码逻辑
     （换顺序跑，先跑的对、后跑的错 3.02），训练里就是每条样本都用着第一条的掩码，
     全程不报错。改成闭在持久缓冲区上才解决。

用法:
    python3 code/abot/probe_action_mask.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "DiffSynth-Studio-h3"))

import torch

from diffsynth.models.minimax_h3_dit import (  # noqa: E402
    _build_action_block_masks, _sdpa_varlen_attention,
)

# 缩小版的真实版面：[图像+锚点 | 37 条标注 | cond | audio | video | pad]
# 比例照搬 7872-cache 实测，只把每帧的行数从 390 缩到 4，好让稠密参考算得动。
LATENT_T, FRAME_ROWS = 6, 4
N_HEAD, ANN_LEN, N_COND, N_AUDIO = 5, 2, 3, 2
N_ANN = LATENT_T * ANN_LEN
V0 = N_HEAD + N_ANN + N_COND + N_AUDIO
USED = V0 + LATENT_T * FRAME_ROWS
SEQ = ((USED + 7) // 8) * 8


def layout():
    ann_rows = torch.tensor(
        [[N_HEAD + k * ANN_LEN, N_HEAD + (k + 1) * ANN_LEN] for k in range(LATENT_T)],
        dtype=torch.long)
    cu = torch.tensor([0, USED, SEQ], dtype=torch.int32)
    return ann_rows, cu


def reference_mask(ann_rows, n_real=None) -> torch.Tensor:
    """参考实现：直接按定义摊成 [USED, USED] 的布尔矩阵。True = 允许。"""
    ann_of = torch.full((SEQ,), -1, dtype=torch.long)
    for k in range(LATENT_T):
        ann_of[ann_rows[k, 0]:ann_rows[k, 1]] = k
    frm_of = torch.full((SEQ,), -1, dtype=torch.long)
    for j in range(LATENT_T):
        frm_of[V0 + j * FRAME_ROWS: V0 + (j + 1) * FRAME_ROWS] = j
    a, f = ann_of[:USED], frm_of[:USED]
    aq, fq = a[:, None], f[:, None]
    akv, fkv = a[None, :], f[None, :]
    blocked = (((aq >= 0) & (fkv >= 0) & (aq != fkv))
               | ((fq >= 0) & (akv >= 0) & (fq != akv)))
    lim = USED if n_real is None else n_real
    idx = torch.arange(USED)
    real = (idx[:, None] < lim) & (idx[None, :] < lim)
    same = idx[:, None] == idx[None, :]
    return same | (real & ~blocked)


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("需要 GPU：FlexAttention 的 create_block_mask 走 CUDA")
    dev = "cuda"
    torch.manual_seed(0)
    ann_rows, cu = layout()
    heads, dim = 2, 16
    q = torch.randn(SEQ, heads, dim, device=dev, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = dim ** -0.5

    fails = []

    # 判据 1：不传掩码 == 改造前
    base = _sdpa_varlen_attention(q, k, v, cu.to(dev), scale, block_masks=None)
    base2 = _sdpa_varlen_attention(q, k, v, cu.to(dev), scale)
    ok = torch.equal(base, base2)
    print(f"  [1] 不传 action_text_rows 与原路径逐位相同 : {'✓' if ok else '✗'}")
    if not ok:
        fails.append(1)

    # 判据 4：与显式布尔掩码的参考实现比
    masks = _build_action_block_masks(ann_rows, V0, FRAME_ROWS, LATENT_T,
                                      cu.to(dev), SEQ, dev)
    got = _sdpa_varlen_attention(q, k, v, cu.to(dev), scale, block_masks=masks)

    allow = reference_mask(ann_rows).to(dev)
    qq = q[:USED].transpose(0, 1).unsqueeze(0)
    kk = k[:USED].transpose(0, 1).unsqueeze(0)
    vv = v[:USED].transpose(0, 1).unsqueeze(0)
    ref_seg = torch.nn.functional.scaled_dot_product_attention(
        qq, kk, vv, attn_mask=allow[None, None], scale=scale)
    ref = base.clone()
    ref[:USED] = ref_seg.squeeze(0).transpose(0, 1)

    err = (got[:USED] - ref[:USED]).abs().max().item()
    ok = err < 2e-4
    print(f"  [4] 与显式布尔掩码参考实现一致        : {'✓' if ok else '✗'}  max|Δ| = {err:.2e}")
    if not ok:
        fails.append(4)

    # 判据 2/3：直接查 allow 矩阵的语义
    ann_of = torch.full((USED,), -1, dtype=torch.long)
    for kk_ in range(LATENT_T):
        ann_of[ann_rows[kk_, 0]:ann_rows[kk_, 1]] = kk_
    frm_of = torch.full((USED,), -1, dtype=torch.long)
    for j in range(LATENT_T):
        frm_of[V0 + j * FRAME_ROWS: V0 + (j + 1) * FRAME_ROWS] = j
    A = allow.cpu()

    same = all(bool(A[i, j]) for i in range(USED) for j in range(USED)
               if ann_of[i] >= 0 and frm_of[j] >= 0 and ann_of[i] == frm_of[j])
    cross = all(not bool(A[i, j]) for i in range(USED) for j in range(USED)
                if ann_of[i] >= 0 and frm_of[j] >= 0 and ann_of[i] != frm_of[j])
    print(f"  [2] 标注 k ↔ 帧 k 放行                : {'✓' if same else '✗'}")
    print(f"      标注 k ↔ 帧 j≠k 屏蔽              : {'✓' if cross else '✗'}")
    if not (same and cross):
        fails.append(2)

    untouched = all(bool(A[i, j]) for i in range(USED) for j in range(USED)
                    if not (ann_of[i] >= 0 and frm_of[j] >= 0)
                    and not (frm_of[i] >= 0 and ann_of[j] >= 0))
    print(f"  [3] 其余全部未被触碰                  : {'✓' if untouched else '✗'}")
    if not untouched:
        fails.append(3)

    # 判据 5：padding 屏蔽
    n_real = USED - 6                       # 末尾 6 行当作 padding
    masks_p = _build_action_block_masks(ann_rows, V0, FRAME_ROWS, LATENT_T,
                                        cu.to(dev), SEQ, dev, n_real=n_real)
    got_p = _sdpa_varlen_attention(q, k, v, cu.to(dev), scale, block_masks=masks_p)
    allow_p = reference_mask(ann_rows, n_real).to(dev)
    ref_p = torch.nn.functional.scaled_dot_product_attention(
        qq, kk, vv, attn_mask=allow_p[None, None], scale=scale)
    err_p = (got_p[:USED] - ref_p.squeeze(0).transpose(0, 1)).abs().max().item()
    Ap = allow_p.cpu()
    no_see_pad = all(not bool(Ap[i, j]) for i in range(n_real) for j in range(n_real, USED))
    pad_self = all(bool(Ap[i, i]) for i in range(n_real, USED))
    pad_only_self = all(not bool(Ap[i, j]) for i in range(n_real, USED)
                        for j in range(USED) if i != j)
    ok5 = err_p < 2e-4 and no_see_pad and pad_self and pad_only_self
    print(f"  [5] padding 行被正确屏蔽              : {'✓' if ok5 else '✗'}  "
          f"max|Δ| = {err_p:.2e}（真实行看不见 pad {no_see_pad}，"
          f"pad 只看自己 {pad_only_self and pad_self}）")
    if not ok5:
        fails.append(5)

    # 判据 6：连续用多个不同的掩码，每个都要对
    errs = []
    for nr in (None, USED - 6, USED - 12, None, USED - 3):
        mk = _build_action_block_masks(ann_rows, V0, FRAME_ROWS, LATENT_T,
                                       cu.to(dev), SEQ, dev, n_real=nr)
        g = _sdpa_varlen_attention(q, k, v, cu.to(dev), scale, block_masks=mk)
        al = reference_mask(ann_rows, nr).to(dev)
        r = torch.nn.functional.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=al[None, None], scale=scale).squeeze(0).transpose(0, 1)
        errs.append((g[:USED] - r).abs().max().item())
    ok6 = max(errs) < 2e-4
    print(f"  [6] 连续 {len(errs)} 个不同掩码全部正确      : {'✓' if ok6 else '✗'}  "
          f"max|Δ| = {max(errs):.2e}")
    if not ok6:
        fails.append(6)

    blocked_frac = 1 - A.float().mean().item()
    print(f"\n  被屏蔽的元素占 {blocked_frac:.3f}"
          f"（版面: {LATENT_T} 帧 × {FRAME_ROWS} 行，标注 {ANN_LEN} 行/条，USED={USED}）")
    print("\n" + ("全部通过 ✓" if not fails else f"失败判据: {fails} ✗"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
