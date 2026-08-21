#!/usr/bin/env python3
"""把 clips/<sid>_w000.npy 的逐帧动作渲染成 HUD，叠在切片视频旁边。

用途是**肉眼验对齐**：按下 W 的那几帧画面该在前进，d_yaw 为正时画面该在转。
`build_abot_clips.py --verify` 的位移扫描 argmin 判据是同一件事的数值版，
但数值版只告诉你"没错一帧"，看不出**动作语义**对不对（比如 COLMAP 反推的
平移方向是否真的和按键一致）。这个脚本补的是后半截。

布局刻意不遮挡画面 —— 视频原样放左边，所有 HUD 在右侧和下方的黑边里：

    ┌──────────────────────┬──────────┐
    │                      │ 键盘     │
    │   视频 832x480       │ 罗盘     │   右栏 400px
    │                      │ 俯视轨迹 │
    ├──────────────────────┴──────────┤
    │ 17 通道 × 130 帧 时间线 + 游标  │   下条 180px
    └─────────────────────────────────┘

底部时间线是整条 clip 的全局视图：一眼能看出动作结构（哪几段在走、
哪几段在转），而不是只看到当前帧。游标标出当前位置。

用法:
    python3 viz_action.py --sample <sample_id>        # 指定样本
    python3 viz_action.py --n 4                       # 从 meta 里取前 4 条
    python3 viz_action.py --n 4 --pick active         # 挑按键最丰富的 4 条
"""
import argparse
import json
import os
import sys

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abot_action as A

ROOT = os.environ.get("ABOT_OUT_ROOT", "/opt/dlami/nvme/danze/minimax_finetune/data")
CLIP_DIR = f"{ROOT}/clips"

VW, VH = 832, 480          # 视频尺寸
SIDE = 400                 # 右栏宽
STRIP = 224                # 下条高（480+224=704，16 的整数倍，免得编码器再缩放）
W, H = VW + SIDE, VH + STRIP

BG = (18, 18, 22)
FG = (225, 225, 230)
DIM = (95, 95, 105)
ON = (70, 220, 200)        # 按键按下
POS = (240, 150, 70)       # 连续量正
NEG = (90, 150, 240)       # 连续量负

# 系统里没有一个字体同时把两边画好：DejaVu 缺 CJK 字形（画成方框），而
# Droid Sans Fallback 是 fallback-only 字体，小字号下拉丁和数字**也**是方框。
# PIL 不做字体回退，所以按 CJK 边界把字符串切段、分别用两个字体画（见 text2）。
_MONO = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono%s.ttf"
_CJK = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"
FONT = (ImageFont.truetype(_MONO % "", 13), ImageFont.truetype(_CJK, 14))
FONT_S = (ImageFont.truetype(_MONO % "", 11), ImageFont.truetype(_CJK, 12))
FONT_B = ImageFont.truetype(_MONO % "-Bold", 15)
FONT_XS = (ImageFont.truetype(_MONO % "", 9), ImageFont.truetype(_CJK, 11))


def _is_cjk(ch):
    return ord(ch) > 0x2E7F


def _runs(t):
    """把字符串切成 (片段, 是否CJK) 的连续段。"""
    out = []
    for ch in t:
        c = _is_cjk(ch)
        if out and out[-1][1] == c:
            out[-1][0] += ch
        else:
            out.append([ch, c])
    return out


def text2(d, xy, t, font, fill, anchor="la"):
    """混排文本：拉丁段用等宽 DejaVu，CJK 段用 Droid，逐段拼接。

    anchor 只支持水平 l/m/r + 垂直 a/m —— HUD 里只用到这几种，
    多余的实现不了也没必要（PIL 的 anchor 是按单次 draw 算的，
    分段之后必须自己先量总宽）。
    """
    mono, cjk = font
    segs = [(s, cjk if c else mono) for s, c in _runs(t)]
    w = sum(f.getlength(s) for s, f in segs)
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    va = "m" if anchor[1] == "m" else "a"
    for s, f in segs:
        d.text((x, y), s, font=f, fill=fill, anchor="l" + va)
        x += f.getlength(s)

# 键盘上的物理排布。IJKL 是视角，和 WASD 分开画，一眼能区分"走"和"看"。
KEYPAD = [
    ("移动", [(None, "W", None), ("A", "S", "D")], 0),
    ("视角", [(None, "I", None), ("J", "K", "L")], 0),
]


def _key_idx(name):
    return A.KEY_COLS.index(name)


def draw_keys(d, x0, y0, act):
    """两个九宫格 + Q/E/Space 一行。按下 = 亮青填充。"""
    cell, gap = 28, 4
    y = y0
    for title, rows, _ in KEYPAD:
        text2(d, (x0, y), title, FONT_S, DIM)
        y += 14
        for row in rows:
            x = x0
            for k in row:
                if k is not None:
                    on = act[_key_idx(k)] > 0.5
                    d.rounded_rectangle([x, y, x + cell, y + cell], 5,
                                        fill=ON if on else None,
                                        outline=ON if on else DIM, width=2)
                    d.text((x + cell / 2, y + cell / 2), k, font=FONT_B,
                           fill=BG if on else DIM, anchor="mm")
                x += cell + gap
            y += cell + gap
        y += 8
    # Q/E/Space —— 数据集里恒零，是天然的阴性对照，画出来是为了让"它一直不亮"
    # 这件事可见，而不是被省略掉看不见。
    x = x0
    for k, wd in (("Q", cell), ("E", cell), ("Space", cell * 3 + gap * 2)):
        on = act[_key_idx(k)] > 0.5
        d.rounded_rectangle([x, y, x + wd, y + cell - 6], 5,
                            fill=ON if on else None,
                            outline=ON if on else DIM, width=2)
        text2(d, (x + wd / 2, y + (cell - 6) / 2), k, FONT_S,
              BG if on else DIM, "mm")
        x += wd + gap
    return y + cell + 6


def draw_compass(d, cx, cy, r, yaw_deg, pitch, roll):
    """罗盘指针 = 累积 yaw 朝向；右侧竖条 = 累积 pitch；圆环倾斜 = 当前 roll。

    用累积量而不是当前帧增量，因为单帧 ~1 度的增量画出来是看不见的抖动；
    累积朝向才和画面里看到的转向对得上。
    """
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=DIM, width=2)
    # 刻度原来标数字，正下方那个 "180" 和 yaw 读数画在同一点，叠成 "1808.0°"。
    # 改画短刻度线、只在正上方留一个 N，读数就没人抢位置。
    for ang in (0, 90, 180, 270):
        a = np.radians(ang - 90)
        d.line([cx + (r - 5) * np.cos(a), cy + (r - 5) * np.sin(a),
                cx + r * np.cos(a), cy + r * np.sin(a)], fill=DIM, width=1)
    text2(d, (cx, cy - r - 9), "N", FONT_XS, DIM, "mm")
    a = np.radians(yaw_deg - 90)
    d.line([cx, cy, cx + r * 0.85 * np.cos(a), cy + r * 0.85 * np.sin(a)],
           fill=POS, width=3)
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=FG)

    bx = cx + r + 34                     # pitch 竖条，中点是水平视线
    d.rectangle([bx, cy - r, bx + 10, cy + r], outline=DIM, width=1)
    frac = float(np.clip(pitch / 45.0, -1, 1))
    yy = cy - frac * r
    d.rectangle([bx, min(cy, yy), bx + 10, max(cy, yy)],
                fill=POS if frac >= 0 else NEG)
    d.line([bx - 3, cy, bx + 13, cy], fill=DIM, width=1)
    text2(d, (bx + 5, cy + r + 10), "pitch", FONT_XS, DIM, "mm")
    # 累积 yaw 转几圈后数字会变长；罗盘表达的是朝向，取模不丢信息，
    # 总转量另起一行（能看出"转了 5 圈"和"转了 8 度"的区别）。
    text2(d, (cx, cy + r + 10), f"yaw {yaw_deg % 360:5.1f}°", FONT_S, DIM, "mm")
    text2(d, (cx, cy + r + 24), f"累计 {yaw_deg:+.0f}°", FONT_S, DIM, "mm")


def draw_path(d, x0, y0, w, h, xz, i):
    """俯视轨迹：COLMAP 反推平移在 x-z 平面的累积路径，红点是当前位置。

    尺度是按 episode 归一化过的（COLMAP 逐 episode 尺度任意，见
    abot_action.py 头注 §2），所以这里只有形状有意义，没有物理单位。
    """
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=DIM, width=1)
    text2(d, (x0 + 4, y0 + 3), "俯视轨迹 (x-z, 归一化)", FONT_S, DIM)
    p = xz - xz.min(0)
    span = max(p.max(), 1e-6)
    pad = 16
    pts = [(x0 + pad + v[0] / span * (w - 2 * pad),
            y0 + h - pad - v[1] / span * (h - 2 * pad - 10)) for v in p]
    if len(pts) > 1:
        d.line(pts, fill=DIM, width=1)
        d.line(pts[:i + 1], fill=ON, width=2)
    cx, cy = pts[i]
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=POS)


def draw_strip(d, x0, y0, w, h, act, i):
    """17 通道 × N 帧 的时间线。

    按键行和相机行给不同行高：按键是二值的，3~4 px 就完全够读；相机增量要看
    幅度大小，行高给足才分得出"轻微偏头"和"猛甩视角"。挤成一样高的话，
    信息量最大的 6 行相机通道反而最难读。
    """
    n = act.shape[0]
    lab_w = 58
    head = 18
    cw = (w - lab_w - 8) / n
    rh_k, rh_c = 10.0, 12.0                      # 按键行 / 相机行
    avail = h - head - 4
    scale = min(1.0, avail / (A.NUM_KEYS * rh_k + 6 * rh_c))
    rh_k, rh_c = rh_k * scale, rh_c * scale

    y = y0 + head
    for c, name in enumerate(A.ACTION_COLS):
        key = c < A.NUM_KEYS
        rh = rh_k if key else rh_c
        v = act[:, c]
        live = (v > 0.5).any() if key else np.abs(v).max() > 1e-6
        text2(d, (x0 + lab_w - 5, y + rh / 2), name, FONT_XS,
              FG if live else DIM, "rm")
        if key:
            for f in np.nonzero(v > 0.5)[0]:
                d.rectangle([x0 + lab_w + f * cw, y,
                             x0 + lab_w + (f + 1) * cw, y + rh - 1], fill=ON)
        else:
            m = max(np.abs(v).max(), 1e-6)
            for f in range(n):
                a = abs(v[f]) / m
                if a < 0.02:
                    continue
                col = POS if v[f] > 0 else NEG
                d.rectangle([x0 + lab_w + f * cw, y,
                             x0 + lab_w + (f + 1) * cw, y + rh - 1],
                            fill=tuple(int(BG[k] + (col[k] - BG[k]) * a)
                                       for k in range(3)))
        y += rh
    cx = x0 + lab_w + (i + 0.5) * cw
    d.line([cx, y0 + head - 3, cx, y], fill=FG, width=1)
    text2(d, (x0 + lab_w, y0 + 2),
          f"帧 {i + 1}/{n}   ← 整条 clip 的动作结构，竖线是当前帧"
          f"　　按键=青　相机增量 正=橙 负=蓝", FONT_XS, DIM)


def render(sid, video_rel, action_rel, out_path, prompt=""):
    frames = imageio.mimread(f"{CLIP_DIR}/{video_rel}", memtest=False)
    act = np.load(f"{CLIP_DIR}/{action_rel}")
    n = min(len(frames), act.shape[0])
    frames, act = frames[:n], act[:n]

    # 累积量：朝向靠 cumsum 出来才看得见（单帧增量 ~1 度是噪声级）。
    cum = np.cumsum(act[:, A.NUM_KEYS:A.NUM_KEYS + 3], axis=0)   # pitch/yaw/roll
    xz = np.cumsum(act[:, [A.NUM_KEYS + 3, A.NUM_KEYS + 5]], axis=0)  # x_right, z_fwd

    wr = imageio.get_writer(out_path, fps=24, codec="libx264",
                            quality=None, ffmpeg_params=["-crf", "18",
                                                         "-pix_fmt", "yuv420p"])
    for i in range(n):
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(Image.fromarray(frames[i]).resize((VW, VH)), (0, 0))
        d = ImageDraw.Draw(canvas)

        # 画面底部可能是天空/沙地这种亮区，白字直接压上去读不出来，垫一条暗衬。
        d.rectangle([0, VH - 26, VW, VH], fill=(0, 0, 0))
        text2(d, (10, VH - 21), f"{sid}  帧 {i + 1}/{n}", FONT, FG)
        x0 = VW + 20
        d.text((x0, 12), "ACTION  17ch", font=FONT_B, fill=FG)
        # 右栏三块必须挤进 VH=480 内，否则会压到底部时间线上。
        # 右栏三块必须挤进 VH=480 内，否则会压到底部时间线上。
        y = draw_keys(d, x0, 32, act[i])
        draw_compass(d, x0 + 74, y + 56, 42, cum[i, 1], cum[i, 0], cum[i, 2])
        py = y + 142
        draw_path(d, x0, py, SIDE - 40, VH - py - 10, xz, i)
        draw_strip(d, 10, VH + 5, W - 20, STRIP - 10, act, i)
        wr.append_data(np.asarray(canvas))
    wr.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=f"{ROOT}/abot_meta_64.jsonl")
    ap.add_argument("--sample", default=None, help="指定 sample_id")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--pick", choices=["head", "active"], default="head",
                    help="active = 按键触发帧数最多的几条，动作最丰富")
    ap.add_argument("--outdir", default=f"{ROOT}/../output/viz")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.meta)]
    if args.sample:
        rows = [r for r in rows if r["sample_id"] == args.sample]
        if not rows:
            sys.exit(f"meta 里没有 sample_id={args.sample}")
    elif args.pick == "active":
        def score(r):
            a = np.load(f"{CLIP_DIR}/{r['action']}")
            return a[:, :A.NUM_KEYS].sum() + np.abs(a[:, A.NUM_KEYS:]).sum()
        rows = sorted(rows, key=score, reverse=True)
    rows = rows[:args.n]

    os.makedirs(args.outdir, exist_ok=True)
    for r in rows:
        out = f"{args.outdir}/{r['sample_id']}_action.mp4"
        n = render(r["sample_id"], r["video"], r["action"], out, r.get("prompt", ""))
        print(f"{r['sample_id']}  {n} 帧  ->  {out}")


if __name__ == "__main__":
    main()
