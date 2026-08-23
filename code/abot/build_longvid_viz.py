#!/usr/bin/env python3
"""长视频两条路的实测对照台：一次生成 10s vs 分块续写 15.5s。

版式沿用标注核对台，动作文本的可视化方式也一样（九宫格按键 HUD + 逐 latent 时间轴），
额外多一条**逐帧变化量**曲线 —— 长视频的两种失败模式（后半段塌掉 / 接缝）
都只能在时间轴上看出来，单看一张截图看不出来。

用法:
    python3 code/abot/build_longvid_viz.py \\
        --out docs/longvid_viz.html --artifact-out .cache/artifact_longvid.html
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abot_action as A  # noqa: E402
import action_script as S  # noqa: E402
import infer_abot as I  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data/clips"
META = ROOT / "data/abot_meta_test_128.jsonl"
INFER = ROOT / "output/abot_inference"
FFMPEG = (ROOT / "../envs/minimax_h3/lib/python3.10/site-packages/imageio_ffmpeg/"
          "binaries/ffmpeg-linux-x86_64-v7.0.2").resolve()
FPS = 24.156
SID = "a3ad9c24bda131dfa0ea18efe44a4e8b"


def encode_video(src: Path, crf: int, width: int) -> str:
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "v.mp4"
        subprocess.run(
            [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(src),
             "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-an", str(dst)],
            check=True)
        return "data:video/mp4;base64," + base64.b64encode(dst.read_bytes()).decode()


def motion_energy(path: Path, w: int = 208, h: int = 120) -> list[float]:
    """逐帧变化量：相邻帧灰度的平均绝对差。塌掉会掉到接近 0，
    硬切会冒出尖峰，接缝会出现一个反常的**低谷**（首尾两帧近乎重复）。"""
    import av
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            frames.append(np.asarray(f.to_image().convert("L").resize((w, h)),
                                     dtype=np.float32))
    g = np.stack(frames)
    return [round(float(x), 3) for x in np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))]


def steps_from_k9(k9: np.ndarray) -> list[dict]:
    script = S.annotate_from_keys9(k9)
    lt = k9.shape[0]
    spans = [e - s for s, e in A.frame_spans(lt)]
    starts = np.concatenate([[0], np.cumsum(spans)[:-1]])
    out = []
    for k in range(lt):
        motion, camera = script[k][len("the man "):].split(", camera ")
        out.append({"t": round(float(starts[k] / FPS), 3), "n": int(spans[k]),
                    "k9": [int(x) for x in k9[k]], "motion": motion,
                    "camera": camera, "text": script[k]})
    return out


def build(crf: int, width: int) -> tuple[list[dict], dict]:
    rows = {json.loads(l)["sample_id"]: json.loads(l) for l in open(META) if l.strip()}
    row = rows[SID]
    items = []

    # ── 路一：一次生成 10s。动作用片段自带的 37 条循环平铺到 72 条 ──
    gen = next((INFER / "long10s_step9840").glob("*/generated.mp4"), None)
    if gen:
        I.set_frames(243)
        _, _, k9, _ = I.load_action(row, CLIPS)
        print("  10s 单次：测变化量 + 编码 …", flush=True)
        items.append({
            "key": "single",
            "title": "路一 · 一次生成 10 秒",
            "note": "num_frames=243，latent_t=72。动作用片段自带的 37 条循环平铺补足。",
            "frames": 243, "latent_t": 72, "seams": [],
            "steps": steps_from_k9(k9),
            "energy": motion_energy(gen),
            "video": encode_video(gen, crf, width),
        })

    # ── 路二：分块续写。每段各自随机动作，段间用尾帧衔接 ──
    joined = INFER / "chunked_step9840/joined.mp4"
    if joined.is_file():
        I.set_frames(124)
        k9 = np.concatenate([I.random_keys9(c) for c in range(3)], axis=0)
        print("  分块续写：测变化量 + 编码 …", flush=True)
        steps = []
        for c in range(3):
            for j, st in enumerate(steps_from_k9(I.random_keys9(c))):
                st = dict(st)
                st["t"] = round(st["t"] + c * 124 / FPS, 3)
                st["chunk"] = c
                steps.append(st)
        items.append({
            "key": "chunked",
            "title": "路二 · 分块续写 15.5 秒",
            "note": "3 段 × 124 帧，上一段尾帧当下一段首帧。每段动作独立随机。",
            "frames": 372, "latent_t": len(steps), "seams": [124, 248],
            "steps": steps,
            "energy": motion_energy(joined),
            "video": encode_video(joined, crf, width),
        })

    for it in items:
        e = np.array(it["energy"])
        it["stats"] = {
            "mean": round(float(e.mean()), 3),
            "cv": round(float(e.std() / e.mean()), 3),
            "quartiles": [round(float(x.mean()), 3) for x in np.array_split(e, 4)],
            "spikes": [[int(i), round(float(e[i]), 2)]
                       for i in np.argsort(e)[-3:][::-1] if e[i] > 3 * e.mean()],
            "seam_ratio": [
                round(float(e[b - 1] / np.concatenate([e[b - 6:b - 1], e[b:b + 5]]).mean()), 2)
                for b in it["seams"]],
        }
    meta = {"sample": SID, "scene": row["prompt"], "keys9": S.KEYS9,
            "checkpoint": "step-9840", "fps": FPS}
    return items, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/longvid_viz.html")
    ap.add_argument("--artifact-out", type=Path, default=None)
    ap.add_argument("--crf", type=int, default=31)
    ap.add_argument("--width", type=int, default=576)
    args = ap.parse_args()

    items, meta = build(args.crf, args.width)
    if not items:
        raise SystemExit("没有可用的长视频结果")
    body = (BODY.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
                .replace("__META__", json.dumps(meta, ensure_ascii=False)))
    for path, text in [
        (args.out, PAGE.replace("__HEAD__", TITLE + FONTS + STYLE).replace("__BODY__", body)),
        (args.artifact_out, TITLE + FONTS + STYLE + body),
    ]:
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"\n写出 {path}  ({path.stat().st_size / 2**20:.1f} MB)")


TITLE = "<title>长视频实测对照台</title>\n"

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500;600&'
         'family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">\n')

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
__HEAD__</head>
<body>
__BODY__</body>
</html>
"""

STYLE = r"""<style>
:root{
  --ground:#EEF1F3; --surface:#FFF; --sunken:#F6F8F9; --line:#D3DAE0;
  --ink:#10151C; --body:#3A444F; --muted:#6E7985;
  --accent:#17A791; --warn:#C56A1E; --bad:#C4423B;
  --c-pan:#17A791; --c-pantilt:#3B72C4; --c-tilt:#7A5AA8;
  --c-follow:#C56A1E; --c-drift:#8A7A1E; --c-static:#9AA3AD;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0B0E13; --surface:#151B24; --sunken:#0F141B; --line:#27313D;
  --ink:#EAEEF2; --body:#B3BDC8; --muted:#79838F;
  --accent:#3ECFB6; --warn:#E68A3C; --bad:#E8706A;
  --c-pan:#3ECFB6; --c-pantilt:#5590E0; --c-tilt:#A98CD8;
  --c-follow:#E68A3C; --c-drift:#C4B44A; --c-static:#5C6672;
}}
:root[data-theme="dark"]{
  --ground:#0B0E13; --surface:#151B24; --sunken:#0F141B; --line:#27313D;
  --ink:#EAEEF2; --body:#B3BDC8; --muted:#79838F;
  --accent:#3ECFB6; --warn:#E68A3C; --bad:#E8706A;
  --c-pan:#3ECFB6; --c-pantilt:#5590E0; --c-tilt:#A98CD8;
  --c-follow:#E68A3C; --c-drift:#C4B44A; --c-static:#5C6672;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--body);font-family:var(--sans);
     font-size:15px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:0 20px 80px}
header{padding:44px 0 12px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.15em;text-transform:uppercase;
         color:var(--muted);margin:0 0 12px}
h1{font-size:clamp(26px,3.4vw,38px);line-height:1.14;font-weight:700;color:var(--ink);
   margin:0 0 14px;letter-spacing:-.02em;text-wrap:balance}
.lede{font-size:17px;max-width:70ch;margin:0 0 22px}
.lede b{color:var(--ink);font-weight:600}
h2{font-size:22px;font-weight:600;color:var(--ink);margin:48px 0 8px;letter-spacing:-.01em}
p{max-width:72ch}
code{font-family:var(--mono);font-size:.87em;background:var(--sunken);border:1px solid var(--line);
     border-radius:3px;padding:1px 5px;color:var(--ink)}
.sub{color:var(--muted);max-width:74ch;margin:0 0 18px}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
       background:var(--line);border:1px solid var(--line);border-radius:7px;
       overflow:hidden;margin:0 0 10px}
.stat{background:var(--surface);padding:14px 17px}
.stat b{display:block;font-family:var(--mono);font-size:20px;color:var(--ink);
        font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stat span{display:block;font-family:var(--mono);font-size:10px;letter-spacing:.08em;
           text-transform:uppercase;color:var(--muted);margin-top:3px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-family:var(--mono);font-size:11.5px;
        color:var(--muted);margin:20px 0 4px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px}

.card{background:var(--surface);border:1px solid var(--line);border-radius:9px;
      margin:22px 0;overflow:hidden}
.card-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 18px;
           border-bottom:1px solid var(--line)}
.sid{font-family:var(--mono);font-size:13.5px;color:var(--ink);font-weight:600}
.chip{font-family:var(--mono);font-size:11px;color:var(--muted);border:1px solid var(--line);
      border-radius:3px;padding:1px 7px}
.chip.bad{color:var(--bad);border-color:var(--bad)}
.chip.ok{color:var(--accent);border-color:var(--accent)}
.spacer{flex:1}
.cnote{font-size:12.5px;color:var(--muted);padding:10px 18px 0}

.stage{max-width:900px;margin:0 auto;padding:16px 18px 4px}
video{width:100%;display:block;border-radius:7px;background:#000}
.strip{display:flex;gap:1px;height:20px;margin-top:12px;border-radius:4px;overflow:hidden;
       border:1px solid var(--line)}
.strip div{cursor:pointer;transition:filter .12s}
.strip div:hover{filter:brightness(1.3)}
.strip div.on{outline:2px solid var(--ink);outline-offset:-2px;z-index:2}
.axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;
      color:var(--muted);margin-top:4px}

/* 逐帧变化量曲线：单条序列，不需要图例，标题已经点明画的是什么。
   接缝用竖线标出，异常尖峰单独打点。 */
.chartbox{margin-top:14px}
.chartbox svg{display:block;width:100%;height:auto}
.ctitle{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
        color:var(--muted);margin-bottom:6px}

.now{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin:14px 0 4px;
     padding:14px 18px;background:var(--sunken);border:1px solid var(--line);border-radius:7px;
     min-height:64px}
.now .txt{font-size:17px;line-height:1.4;color:var(--body);flex:1;min-width:260px}
.now .txt b{color:var(--ink);font-weight:600}
.now .k{font-family:var(--mono);font-size:11px;color:var(--muted);white-space:nowrap}

.pads{display:flex;gap:14px;align-items:flex-end}
.pad{display:flex;flex-direction:column;gap:4px}
.pad .lbl{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;color:var(--muted);
          text-transform:uppercase}
.pad .r{display:flex;gap:4px}
.pad .k{width:28px;height:28px;border-radius:5px;border:2px solid var(--line);
        font-family:var(--mono);font-size:12px;font-weight:600;color:var(--muted);
        display:flex;align-items:center;justify-content:center}
.pad .k.on{background:var(--accent);border-color:var(--accent);color:#fff}
.pad .k.hole{border-color:transparent}
.pad .k.wide{width:96px;height:20px;font-size:10px;border-radius:4px}
.pad.sm{gap:2px} .pad.sm .lbl{display:none} .pad.sm .r{gap:2px}
.pad.sm .k{width:12px;height:12px;border-width:1.5px;border-radius:3px;font-size:0}
.pad.sm .k.wide{width:40px;height:9px}

.rows{padding:6px 0 12px;max-height:264px;overflow-y:auto;border-top:1px solid var(--line)}
.row{display:grid;grid-template-columns:78px 176px 1fr;gap:12px;align-items:center;
     padding:5px 22px;font-size:13.5px;border-left:3px solid transparent;cursor:pointer}
.row:hover{background:var(--sunken)}
.row.on{background:var(--sunken);border-left-color:var(--ink)}
.row .k{font-family:var(--mono);font-size:11px;color:var(--muted);
        font-variant-numeric:tabular-nums}
.row .txt{color:var(--body)}
.row .txt b{color:var(--ink);font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:7px;
     vertical-align:baseline}

.scroll{overflow-x:auto;margin:18px 0}
table{width:100%;border-collapse:collapse;font-size:14px;min-width:640px;
      background:var(--surface);border:1px solid var(--line);border-radius:7px}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-family:var(--mono);font-size:10.5px;font-weight:500;letter-spacing:.08em;
         text-transform:uppercase;color:var(--muted);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
td.nm{font-family:var(--mono);font-size:12.5px;color:var(--ink);white-space:nowrap}
td.num{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums;
       text-align:right;color:var(--ink);white-space:nowrap}
.note{background:var(--sunken);border:1px solid var(--line);border-left:3px solid var(--accent);
      border-radius:0 7px 7px 0;padding:14px 18px;margin:20px 0}
.note p{margin:0;max-width:none}
.note.warn{border-left-color:var(--warn)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
"""

BODY = r"""<div class="wrap">
<header>
  <p class="eyebrow">MiniMax-H3 · ABot 逐 latent 文本注入</p>
  <h1>长视频实测对照台</h1>
  <p class="lede">当前 checkpoint 只训过 5.2 秒。两条不改架构、不重训的路各跑了一条实测：
  <b>一次生成 10 秒</b>，和<b>分块续写 15.5 秒</b>。
  下面每条都配了逐 latent 的动作文本（和其它核对台一样的九宫格 HUD），
  外加一条<b>逐帧变化量曲线</b> —— 长视频的两种失败模式都只能在时间轴上看出来。</p>
  <div class="stats" id="stats"></div>
  <div class="legend" id="legend"></div>
</header>

<div id="cards"></div>

<h2>两条路的实测数字</h2>
<div class="scroll"><table id="tbl"></table></div>

<div class="note warn">
<p><b>两个失败模式都抓到了，而且分块那个和预想的方向相反。</b>
一次生成 10 秒<b>没有</b>塌掉（后半段的运动量甚至更高），但稳定性差得多，
中间冒出几帧数倍于均值的硬切。分块续写整体平稳，接缝处却是一个反常的<b>低谷</b> ——
变化量只有邻域的一半，因为下一段的首帧就是上一段的尾帧，两帧近乎重复，
看上去就是"动作卡了一下"。这正是"只传了一帧、传不了运动状态"的直接后果。</p>
</div>

<div class="note">
<p><b>这页判不了什么。</b>它量的是<b>时间上的连续性</b>，不是画面质量，也不是动作服从度。
一条画面糊掉但运动平稳的视频，在这里的数字会很好看。
每条路各只跑了一条、一个 seed，是存在性证据不是统计结论。</p>
</div>
</div>
<script>
// 所有 const 必须声明在使用它们的代码之前：函数声明会提升、const 不会，
// 顺序错了整个 script 抛 ReferenceError，页面一片空白，而静态检查全是绿的。
const ITEMS = __ITEMS__, META = __META__;
const KEYS9 = META.keys9;
const KIDX = Object.fromEntries(KEYS9.map((n, i) => [n, i]));
const PADS = [["移动", [[null,"W",null],["A","S","D"]]], ["视角", [[null,"I",null],["J","K","L"]]]];
const CATNAME = {pan:"摇镜", pantilt:"摇镜+俯仰", tilt:"俯仰",
                 follow:"跟随", drift:"平移", static:"静止"};
const CATORDER = ["pan","pantilt","tilt","follow","drift","static"];
const cv = c => `var(--c-${c})`;
const esc = s => String(s).replace(/[&<>"]/g, m =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[m]));

function catOf(camera){
  if (camera === "holds steady") return "static";
  if (camera === "follows him") return "follow";
  if (camera.indexOf("drifts") >= 0) return "drift";
  const p = camera.indexOf("pans") >= 0, t = camera.indexOf("tilts") >= 0;
  return p && t ? "pantilt" : p ? "pan" : "tilt";
}

function padsHtml(k9, sm){
  const cls = sm ? "pad sm" : "pad";
  const cell = n => n === null
    ? '<div class="k hole"></div>'
    : `<div class="k ${k9[KIDX[n]] ? "on" : ""}">${sm ? "" : n}</div>`;
  const pads = PADS.map(([lbl, rows]) => `<div class="${cls}">
      <span class="lbl">${lbl}</span>
      ${rows.map(r => `<div class="r">${r.map(cell).join("")}</div>`).join("")}</div>`);
  pads.push(`<div class="${cls}"><span class="lbl">快</span>
      <div class="r"><div class="k ${sm ? "" : "wide "}${k9[KIDX.F] ? "on" : ""}">${sm ? "" : "FAST"}</div></div></div>`);
  return `<div class="pads">${pads.join("")}</div>`;
}

// 逐帧变化量曲线。单条序列，所以不放图例 —— 标题已经点明画的是什么。
function chartSvg(it){
  const e = it.energy, n = e.length;
  const W = 900, H = 132, L = 44, R = 12, T = 10, B = 22;
  const max = Math.max(...e) * 1.06;
  const x = i => L + i / (n - 1) * (W - L - R);
  const y = v => T + (1 - v / max) * (H - T - B);
  const pts = e.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const seams = it.seams.map(b => `
    <line x1="${x(b - 1).toFixed(1)}" y1="${T}" x2="${x(b - 1).toFixed(1)}" y2="${H - B}"
          stroke="var(--warn)" stroke-width="1.4" stroke-dasharray="4 3"/>
    <text x="${x(b - 1).toFixed(1)}" y="${T + 10}" font-size="10" fill="var(--warn)"
          text-anchor="middle">接缝</text>
    <circle cx="${x(b - 1).toFixed(1)}" cy="${y(e[b - 1]).toFixed(1)}" r="3.5"
            fill="var(--warn)"/>`).join("");
  const spikes = (it.stats.spikes || []).map(([i, v]) => `
    <circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="3.5" fill="var(--bad)"/>
    <text x="${x(i).toFixed(1)}" y="${(y(v) - 7).toFixed(1)}" font-size="9.5"
          fill="var(--bad)" text-anchor="middle">${v}</text>`).join("");
  const mean = it.stats.mean;
  return `<svg viewBox="0 0 ${W} ${H}" role="img"
      aria-label="${esc(it.title)}的逐帧变化量曲线，均值 ${mean}，变异系数 ${it.stats.cv}">
    <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="currentColor"
          stroke-width="1" opacity=".3"/>
    <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="currentColor"
          stroke-width="1" opacity=".3"/>
    <line x1="${L}" y1="${y(mean).toFixed(1)}" x2="${W - R}" y2="${y(mean).toFixed(1)}"
          stroke="currentColor" stroke-width="1" stroke-dasharray="3 4" opacity=".4"/>
    <text x="${L - 6}" y="${(y(mean) + 3).toFixed(1)}" font-size="9.5" fill="currentColor"
          opacity=".55" text-anchor="end">均值 ${mean}</text>
    <text x="${L - 6}" y="${T + 8}" font-size="9.5" fill="currentColor" opacity=".55"
          text-anchor="end">${max.toFixed(0)}</text>
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.6"
              stroke-linejoin="round"/>
    ${seams}${spikes}
    <text x="${L}" y="${H - 6}" font-size="10" fill="currentColor" opacity=".55">0s</text>
    <text x="${W - R}" y="${H - 6}" font-size="10" fill="currentColor" opacity=".55"
          text-anchor="end">${(n / META.fps).toFixed(1)}s</text>
  </svg>`;
}

function renderStats(){
  const s = ITEMS.find(x => x.key === "single"), c = ITEMS.find(x => x.key === "chunked");
  document.getElementById("stats").innerHTML = [
    [META.checkpoint, "checkpoint（只训过 5.2s）"],
    [s ? s.frames + " 帧" : "—", "一次生成"],
    [c ? c.frames + " 帧" : "—", "分块续写"],
    [s ? s.stats.cv : "—", "一次生成 变异系数"],
    [c ? c.stats.cv : "—", "分块 变异系数"],
  ].map(([b, t]) => `<div class="stat"><b>${esc(b)}</b><span>${esc(t)}</span></div>`).join("");
  document.getElementById("legend").innerHTML = CATORDER
    .map(k => `<span><i style="background:${cv(k)}"></i>${CATNAME[k]}</span>`).join("");
}

function renderCards(){
  document.getElementById("cards").innerHTML = ITEMS.map((it, ci) => {
    const strip = it.steps.map((st, k) =>
      `<div data-c="${ci}" data-k="${k}" title="latent ${k} · ${esc(st.text)}"
            style="flex:${st.n};background:${cv(catOf(st.camera))}"></div>`).join("");
    const rows = it.steps.map((st, k) => `<div class="row" data-c="${ci}" data-k="${k}">
        <span class="k">${k} · ${st.t.toFixed(2)}s</span>
        ${padsHtml(st.k9, true)}
        <span class="txt"><i class="dot" style="background:${cv(catOf(st.camera))}"></i>the man
          <b>${esc(st.motion)}</b>, camera <b>${esc(st.camera)}</b></span></div>`).join("");
    const q = it.stats.quartiles.map(v => v.toFixed(2)).join(" → ");
    const cvChip = it.stats.cv > 0.7
      ? `<span class="chip bad">变异系数 ${it.stats.cv}（抖）</span>`
      : `<span class="chip ok">变异系数 ${it.stats.cv}（稳）</span>`;
    return `<div class="card">
      <div class="card-head">
        <span class="sid">${esc(it.title)}</span>
        <span class="chip">${it.frames} 帧 · ${(it.frames / META.fps).toFixed(1)}s</span>
        <span class="chip">latent ${it.latent_t}</span>
        <span class="spacer"></span>
        ${cvChip}
        <span class="chip">四分段 ${q}</span>
      </div>
      <p class="cnote">${esc(it.note)}</p>
      <div class="stage">
        <video src="${it.video}" controls loop muted playsinline data-c="${ci}"></video>
        <div class="strip" data-c="${ci}">${strip}</div>
        <div class="axis"><span>0.00s</span><span>动作文本时间轴（latent 0 → ${it.latent_t - 1}）</span>
          <span>${(it.frames / META.fps).toFixed(2)}s</span></div>
        <div class="chartbox">
          <div class="ctitle">逐帧变化量（相邻帧灰度平均绝对差）</div>
          ${chartSvg(it)}
        </div>
        <div class="now" id="now-${ci}"></div>
      </div>
      <div class="rows" data-c="${ci}">${rows}</div>
    </div>`;
  }).join("");
}

function highlight(ci, k){
  const it = ITEMS[ci], st = it.steps[k];
  const now = document.getElementById(`now-${ci}`);
  if (now) now.innerHTML = `${padsHtml(st.k9)}
    <span class="txt"><i class="dot" style="background:${cv(catOf(st.camera))}"></i>the man
      <b>${esc(st.motion)}</b>, camera <b>${esc(st.camera)}</b></span>
    <span class="k">latent ${k} · ${st.t.toFixed(2)}s${
      st.chunk !== undefined ? ` · 第 ${st.chunk + 1} 段` : ""}</span>`;
  document.querySelectorAll(`.strip[data-c="${ci}"] > div`).forEach((d, i) =>
    d.classList.toggle("on", i === k));
  const rows = document.querySelectorAll(`.rows[data-c="${ci}"] .row`);
  rows.forEach((r, i) => r.classList.toggle("on", i === k));
  const row = rows[k], box = document.querySelector(`.rows[data-c="${ci}"]`);
  if (row && box){
    const top = row.offsetTop - box.offsetTop, h = box.clientHeight;
    if (top < box.scrollTop || top > box.scrollTop + h - row.clientHeight - 8)
      box.scrollTo({top: top - h / 2, behavior: "smooth"});
  }
}
function stepAt(ci, t){
  const st = ITEMS[ci].steps;
  let k = 0;
  while (k + 1 < st.length && st[k + 1].t <= t) k++;
  return k;
}

function renderTable(){
  document.getElementById("tbl").innerHTML =
    `<thead><tr><th>指标</th>${ITEMS.map(i => `<th>${esc(i.title)}</th>`).join("")}
      <th>怎么读</th></tr></thead><tbody>` +
    [["长度", i => `${i.frames} 帧 = ${(i.frames / META.fps).toFixed(1)}s`, "—"],
     ["latent 数", i => i.latent_t, "每条对应一句动作文本"],
     ["变化量均值", i => i.stats.mean, "整体运动强度"],
     ["变异系数", i => i.stats.cv, "<b>越小越平稳</b>；大说明忽快忽慢"],
     ["四分段均值", i => i.stats.quartiles.join(" / "),
      "后段掉到接近 0 = 塌了。两条都<b>没有</b>"],
     ["异常尖峰", i => (i.stats.spikes.length ? i.stats.spikes.map(
        s => `第${s[0]}帧 ${s[1]}`).join("、") : "无"), "数倍于均值 = 硬切"],
     ["接缝比值", i => (i.seam_ratio_txt = i.stats.seam_ratio.length
        ? i.stats.seam_ratio.join(" / ") : "不适用"),
      "接缝处变化量 ÷ 邻域。<b>远小于 1 = 动作卡住</b>"],
    ].map(([nm, fn, how]) => `<tr><td class="nm">${nm}</td>` +
      ITEMS.map(i => `<td class="num">${fn(i)}</td>`).join("") +
      `<td>${how}</td></tr>`).join("") + `</tbody>`;
}

renderStats();
renderCards();
renderTable();
ITEMS.forEach((_, ci) => highlight(ci, 0));
document.querySelectorAll("video").forEach(v => {
  const ci = +v.dataset.c;
  v.addEventListener("timeupdate", () => highlight(ci, stepAt(ci, v.currentTime)));
});
document.addEventListener("click", e => {
  const el = e.target.closest("[data-k]");
  if (!el) return;
  const ci = +el.dataset.c, k = +el.dataset.k;
  const v = document.querySelector(`video[data-c="${ci}"]`);
  if (v){ v.currentTime = ITEMS[ci].steps[k].t; highlight(ci, k); }
});
</script>
"""


if __name__ == "__main__":
    main()
