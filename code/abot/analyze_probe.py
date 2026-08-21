#!/usr/bin/env python3
"""探针判据：文本尾句有没有真的改变画面的相机运动。

不靠肉眼。用相位相关估每对相邻帧的全局水平/垂直位移，累加成整段的净平移。

**关键点：变体之间同 seed 同首帧，所以任何差异都归因于 prompt。**
但"有差异"本身不算数 —— 换了文本嵌入，去噪轨迹必然有扰动。真正的判据是**方向性**，
而且必须相对 none 基线来看，不能看绝对符号 —— 片段本身就有固有运动，基线不是零。

    d_left  = cum_dx(left)  − cum_dx(none)   应为负（更向左）
    d_right = cum_dx(right) − cum_dx(none)   应为正（更向右）
    效应量与 none/still 这对语义相近变体之间的差做比较

符号约定已在 128 条测试集的前 14 条上标定：cum_dx 与 npy 的 Σd_yaw
相关系数 r = −0.936，14/14 反号。J=左转→d_yaw 正，因此 **cum_dx 为负 = 相机左摇**。

用法:
    python3 code/abot/analyze_probe.py --dir output/abot_inference/probe_text
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_gray(path: Path, max_frames: int = 124, w: int = 208, h: int = 120):
    """读成 [T, h, w] 的灰度 float32，降采样只为提速，不影响位移估计的符号。"""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), "ffmpeg")
    out = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        a = np.asarray(frame, dtype=np.float32)
        if a.ndim == 3:
            a = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        sh, sw = a.shape[0] // h, a.shape[1] // w
        a = a[: h * sh, : w * sw].reshape(h, sh, w, sw).mean(axis=(1, 3))
        out.append(a)
    reader.close()
    return np.stack(out)


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """相位相关：返回把 a 对齐到 b 所需的 (dy, dx)，单位是降采样后的像素。"""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    mag = np.abs(r)
    r = r / np.where(mag < 1e-8, 1e-8, mag)
    corr = np.fft.ifft2(r).real
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = float(idx[0]), float(idx[1])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dy, dx


def track(path: Path) -> dict:
    g = read_gray(path)
    dys, dxs = [], []
    for t in range(len(g) - 1):
        dy, dx = phase_shift(g[t], g[t + 1])
        dys.append(dy)
        dxs.append(dx)
    dxs, dys = np.array(dxs), np.array(dys)
    return {
        "frames": len(g),
        "cum_dx": float(dxs.sum()),
        "cum_dy": float(dys.sum()),
        "mean_abs_dx": float(np.abs(dxs).mean()),
        "gray": g,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()

    # 跳过 write_video_atomic 的中间产物，否则会读到写了一半又被改名的文件
    vids = sorted(p for p in args.dir.glob("*.mp4") if ".tmp." not in p.name)
    if not vids:
        raise SystemExit(f"没有 mp4: {args.dir}")

    res = {}
    for p in vids:
        res[p.stem] = track(p)
        print(f"  读完 {p.stem}", flush=True)

    print("\n每个变体的全局平移（相位相关，降采样 208×120 像素单位）")
    print(f"{'变体':<18}{'累计水平':>10}{'累计垂直':>10}{'逐帧|dx|均值':>14}")
    for k in sorted(res):
        r = res[k]
        print(f"{k:<18}{r['cum_dx']:>10.1f}{r['cum_dy']:>10.1f}{r['mean_abs_dx']:>14.2f}")

    def pair_mae(a: str, b: str) -> float:
        ga, gb = res[a]["gray"], res[b]["gray"]
        n = min(len(ga), len(gb))
        return float(np.abs(ga[:n] - gb[:n]).mean())

    print("\n变体两两之间的逐帧像素差（0–255 尺度，同 seed 同首帧）")
    keys = sorted(res)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a.rsplit("_cfg", 1)[-1] != b.rsplit("_cfg", 1)[-1]:
                continue  # 只比同一档 cfg
            print(f"  {a:<18} vs {b:<18}  MAE {pair_mae(a, b):6.2f}")

    print("\n判据（cum_dx 为负 = 相机左摇，标定 r=-0.936）")
    verdicts = []
    for cfg in sorted({k.rsplit("_cfg", 1)[-1] for k in keys}):
        need = [f"{v}_cfg{cfg}" for v in ("none", "left", "right")]
        if any(n not in res for n in need):
            continue
        base = res[f"none_cfg{cfg}"]["cum_dx"]
        dl = res[f"left_cfg{cfg}"]["cum_dx"] - base
        dr = res[f"right_cfg{cfg}"]["cum_dx"] - base
        ok_l, ok_r = dl < 0, dr > 0
        null = (pair_mae(f"none_cfg{cfg}", f"still_cfg{cfg}")
                if f"still_cfg{cfg}" in res else None)
        sig = pair_mae(f"left_cfg{cfg}", f"right_cfg{cfg}")
        ratio = (sig / null) if null else float("nan")
        print(f"  cfg={cfg}  基线 none = {base:+.0f}")
        print(f"    left  相对基线 {dl:+7.1f}  {'✓ 更向左' if ok_l else '✗ 方向错'}")
        print(f"    right 相对基线 {dr:+7.1f}  {'✓ 更向右' if ok_r else '✗ 方向错'}")
        print(f"    效应量 left-right MAE {sig:.2f} / none-still 空对照 "
              f"{null:.2f} = {ratio:.2f}×" if null else
              f"    效应量 left-right MAE {sig:.2f}")
        order = [res[f"{v}_cfg{cfg}"]["mean_abs_dx"] for v in ("still", "right", "none", "left")
                 if f"{v}_cfg{cfg}" in res]
        print(f"    逐帧|dx| 按 still<right<none<left 排序: "
              f"{[round(v,2) for v in order]}  "
              f"{'✓ 单调' if order == sorted(order) else '✗ 不单调'}")
        verdicts.append((ok_l and ok_r, ratio, order == sorted(order)))

    print()
    strong = [v for v in verdicts if v[0] and (v[1] != v[1] or v[1] > 2.0)]
    if strong:
        print("结论: ✅ 文本通道对相机运动有方向性控制 —— 逐 latent 文本注入值得往下做")
    elif any(v[0] for v in verdicts):
        print("结论: ⚠️ 方向对但效应量小 —— 通道活着但弱，需要 CFG 放大 + 训练")
    else:
        print("结论: ❌ 文本通道对运动无方向性控制 —— 该回头走 PRoPE 那条路")

    out = {k: {kk: vv for kk, vv in v.items() if kk != "gray"} for k, v in res.items()}
    (args.dir / "probe_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指标已写入 {args.dir / 'probe_metrics.json'}")


if __name__ == "__main__":
    main()
