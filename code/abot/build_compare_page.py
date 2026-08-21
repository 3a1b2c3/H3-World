#!/usr/bin/env python3
"""把 infer_abot.py 的产物做成 GT / Generated 横版对比页。

和 `viz_action.py` 的分工：那个把 HUD **渲进视频**（离线看单条切片用），
这里把 HUD 做成页面上的 JS 层，跟着播放进度走。选后者是因为对比场景要同时
播两条视频、还要能拖动比对，渲进视频就没法交互，而且两条视频各渲一遍
体积翻倍。视觉语言沿用同一套：按键=青，相机正=橙负=蓝。

关键的一条信息设计：**8 个按键是喂给模型的控制输入，6 个相机通道不是**
（`conditioning: first_frame+8d_actions`）。相机通道画出来是当参照物——
指令说 D（右移），GT 的 d_x 确实在动，那就可以直接看 generated 有没有跟着动。
两者不区分开画的话，很容易误以为相机增量也是输入。

用法:
    python3 build_compare_page.py                       # 取最新一次推理
    python3 build_compare_page.py --run step500_gpu7_1sample
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abot_action as A

ROOT = Path(__file__).resolve().parents[2]
INFER_ROOT = ROOT / "output" / "abot_inference"
CLIP_ROOT = ROOT / "data" / "clips"


def runs_to_spans(v: np.ndarray) -> list[list[int]]:
    """二值序列 -> [[起,止), ...]。按键在时间上是成段的，存区间比存 124 个 0/1
    小一个量级，页面里也更好画。"""
    out, start = [], None
    for i, on in enumerate(v > 0.5):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append([start, i]); start = None
    if start is not None:
        out.append([start, len(v)])
    return out


def build_sample(s: dict, run_dir: Path, n_frames: int) -> dict | None:
    if s.get("status") != "complete":
        return None
    act_path = CLIP_ROOT / s["source_action"]
    if not act_path.exists():
        print(f"  ! 缺动作文件，跳过: {act_path}")
        return None
    a = np.load(act_path)[:n_frames]

    keys = {name: runs_to_spans(a[:, idx])
            for name, idx in zip(A.ACTIVE_KEY_COLS, A.ACTIVE_KEY_INDICES)}
    # 相机通道量化成 -100..100 的整数：页面只拿它画色条深浅，float 全精度是浪费。
    cam_raw = a[:, A.NUM_KEYS:]
    scale = max(float(np.abs(cam_raw).max()), 1e-6)
    cam = {A.ACTION_COLS[A.NUM_KEYS + i]: [int(round(x * 100 / scale)) for x in cam_raw[:, i]]
           for i in range(cam_raw.shape[1])}

    sid = s["sample_id"]
    return {
        "id": sid,
        "short": sid[:8],
        "seed": s.get("seed", 0),
        "prompt": s.get("prompt", ""),
        "gt": s["gt_video"],
        "gen": s["generated_video"],
        "frames": n_frames,
        "keys": keys,
        "cam": cam,
        "camScale": round(scale, 3),
        "keyTotals": {k: int(sum(e - b for b, e in v)) for k, v in keys.items()},
        "mode": action_mode(keys),
    }


MOVE_KEYS, LOOK_KEYS = {"W", "A", "S", "D"}, {"I", "J", "K", "L"}


def action_mode(keys: dict) -> str:
    """按下过的键落在哪一组。纯视角的样本最有诊断价值——没有位移干扰，
    画面转动只可能来自视角指令。"""
    live = {k for k, v in keys.items() if v}
    mv, lk = live & MOVE_KEYS, live & LOOK_KEYS
    if mv and lk:
        return "移动+视角"
    if lk:
        return "纯视角"
    if mv:
        return "纯移动"
    return "无输入"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="output/abot_inference 下的子目录名")
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--embed-from", default=None, metavar="DIR",
                    help="把视频以 data URI 内嵌（用该目录下的同名压缩版）。"
                         "本地看不需要——这是为了发成单文件 Artifact，"
                         "那边的 CSP 不允许引用外部文件。")
    args = ap.parse_args()

    if args.run:
        run_dir = INFER_ROOT / args.run
    else:
        cands = sorted((d for d in INFER_ROOT.iterdir() if (d / "manifest.json").exists()),
                       key=lambda d: (d / "manifest.json").stat().st_mtime)
        if not cands:
            sys.exit(f"没找到任何 manifest.json: {INFER_ROOT}")
        run_dir = cands[-1]
    man = json.loads((run_dir / "manifest.json").read_text())
    cfg, ck = man["config"], man["checkpoint"]
    n_frames = int(cfg.get("num_frames", 124))

    print(f"run       : {run_dir.name}")
    samples = [x for x in (build_sample(s, run_dir, n_frames) for s in man["samples"]) if x]
    print(f"样本      : {len(samples)} / {len(man['samples'])}")
    if not samples:
        sys.exit("没有可用样本")

    ckpt_name = Path(ck["path"]).name
    train_n = Path(ck["path"]).parent.name
    meta = {
        "run": run_dir.name,
        "ckpt": ckpt_name,
        "trainN": train_n,
        "actionDim": ck.get("action_dim"),
        "cols": cfg.get("action_columns", A.ACTIVE_KEY_COLS),
        "cond": cfg.get("conditioning", ""),
        "steps": cfg.get("steps"),
        "fps": cfg.get("fps", 24),
        "size": f'{cfg.get("width")}×{cfg.get("height")}',
        "created": man.get("created_at", "")[:19].replace("T", " "),
    }

    if args.embed_from:
        import base64
        src_root = Path(args.embed_from)
        total = 0
        for sm in samples:
            for key in ("gt", "gen"):
                f = src_root / sm[key]
                if not f.exists():
                    sys.exit(f"内嵌源缺文件: {f}")
                b = f.read_bytes(); total += len(b)
                sm[key] = "data:video/mp4;base64," + base64.b64encode(b).decode()
        print(f"内嵌      : 12 个视频 {total/2**20:.1f} MB -> base64 {total*1.33/2**20:.1f} MB")

    html = PAGE.replace("__META__", json.dumps(meta, ensure_ascii=False)) \
               .replace("__SAMPLES__", json.dumps(samples, ensure_ascii=False)) \
               .replace("__CAMCOLS__", json.dumps(A.ACTION_COLS[A.NUM_KEYS:], ensure_ascii=False))
    out = run_dir / args.out
    if out.exists() and args.out == "index.html":
        bak = run_dir / "index_v1.html"
        if not bak.exists():
            bak.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"原页面已备份 -> {bak.name}")
    out.write_text(html, encoding="utf-8")
    print(f"页面      : {out}  ({out.stat().st_size/1024:.0f} KB)")


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ABot 动作条件对比</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --ground:#EEF1F3; --surface:#FFF; --sunken:#F6F8F9; --line:#D3DAE0;
  --ink:#10151C; --body:#3A444F; --muted:#6E7985;
  --signal:#17A791; --warm:#C56A1E; --cool:#3B72C4;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0B0E13; --surface:#151B24; --sunken:#0F141B; --line:#27313D;
  --ink:#EAEEF2; --body:#B3BDC8; --muted:#79838F;
  --signal:#3ECFB6; --warm:#E68A3C; --cool:#5590E0;
}}
:root[data-theme="dark"]{
  --ground:#0B0E13; --surface:#151B24; --sunken:#0F141B; --line:#27313D;
  --ink:#EAEEF2; --body:#B3BDC8; --muted:#79838F;
  --signal:#3ECFB6; --warm:#E68A3C; --cool:#5590E0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--body);font-family:var(--sans);
     font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1760px;margin:0 auto;padding:0 20px 72px}
header{padding:34px 0 18px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
         color:var(--muted);margin:0 0 12px}
h1{font-size:clamp(24px,3vw,34px);line-height:1.15;font-weight:700;color:var(--ink);
   margin:0 0 14px;letter-spacing:-.02em}
.lede{max-width:76ch;margin:0 0 22px}
.lede b{color:var(--ink);font-weight:600}
.meta{display:flex;flex-wrap:wrap;gap:1px;background:var(--line);border:1px solid var(--line);
      border-radius:6px;overflow:hidden;margin-bottom:10px}
.meta div{background:var(--surface);padding:11px 16px;flex:1 1 128px}
.meta b{display:block;font-family:var(--mono);font-size:15px;color:var(--ink);
        font-variant-numeric:tabular-nums}
.meta span{display:block;font-family:var(--mono);font-size:10px;letter-spacing:.08em;
           text-transform:uppercase;color:var(--muted);margin-top:3px}
.band{display:flex;gap:2px;height:3px;margin:26px 0 22px}
.band i{flex:1;border-radius:1px;opacity:.75}

.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;
      margin-bottom:22px;overflow:hidden}
.card-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;
           padding:13px 18px;border-bottom:1px solid var(--line)}
.idx{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--signal)}
.sid{font-family:var(--mono);font-size:13px;color:var(--ink)}
.chip{font-family:var(--mono);font-size:11px;color:var(--muted);
      border:1px solid var(--line);border-radius:3px;padding:1px 7px}
.spacer{flex:1}
.frame-ind{font-family:var(--mono);font-size:12px;color:var(--muted);
           font-variant-numeric:tabular-nums}

.stage{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
.pane{background:var(--sunken);position:relative}
.pane-tag{position:absolute;top:9px;left:9px;z-index:2;font-family:var(--mono);font-size:11px;
          font-weight:600;letter-spacing:.08em;padding:2px 9px;border-radius:3px;
          background:rgba(0,0,0,.62);color:#fff}
.pane.gen .pane-tag{background:var(--signal);color:#04120F}
video{display:block;width:100%;aspect-ratio:832/480;background:#000;object-fit:contain}

.hud{display:grid;grid-template-columns:212px 1fr;gap:18px;padding:15px 18px 17px;
     border-top:1px solid var(--line)}
.keys{display:flex;flex-direction:column;gap:9px}
.krow{display:flex;gap:5px;justify-content:center}
.klabel{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;
        color:var(--muted);text-align:center}
.key{width:34px;height:30px;border:1.5px solid var(--line);border-radius:5px;
     display:flex;align-items:center;justify-content:center;font-family:var(--mono);
     font-size:12px;font-weight:600;color:var(--muted);transition:none}
.key.on{background:var(--signal);border-color:var(--signal);color:#04120F}
.key.void{opacity:.28}

.tl{position:relative;min-width:0}
.tl canvas{display:block;width:100%;height:auto;cursor:pointer}
.cursor{position:absolute;top:0;bottom:0;width:1px;background:var(--ink);pointer-events:none;
        opacity:.85}
.tl-legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:7px;font-family:var(--mono);
           font-size:10px;color:var(--muted)}
.sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;
    vertical-align:baseline}

.foot{display:flex;align-items:center;gap:16px;flex-wrap:wrap;
      padding:10px 18px 13px;border-top:1px solid var(--line);font-size:13px}
.foot label{display:inline-flex;align-items:center;gap:6px;color:var(--muted);
            font-family:var(--mono);font-size:11px;cursor:pointer}
details{padding:0 18px 14px}
summary{cursor:pointer;font-family:var(--mono);font-size:11px;color:var(--muted)}
details p{max-width:96ch;margin:9px 0 0;font-size:13px;color:var(--body)}
.toolbar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:0 0 18px}
.toolbar button{font-family:var(--mono);font-size:12px;color:var(--ink);background:var(--surface);
  border:1px solid var(--line);border-radius:5px;padding:6px 13px;cursor:pointer}
.toolbar button:hover{border-color:var(--signal);color:var(--signal)}
.hint{font-family:var(--mono);font-size:11px;color:var(--muted)}
.chip.mode{color:var(--signal);border-color:currentColor}
.note{border-left:2px solid var(--warm);background:var(--surface);padding:13px 17px;
      margin:0 0 22px;border-radius:0 6px 6px 0;font-size:14px}
.note b{color:var(--ink)}
footer{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);
       font-family:var(--mono);font-size:11px;color:var(--muted)}
:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
@media(max-width:1080px){.stage{grid-template-columns:1fr}.hud{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<header>
  <p class="eyebrow" id="eyebrow"></p>
  <h1>动作指令有没有真的控制住生成</h1>
  <p class="lede">左边是真实切片，右边是模型在<b>同一个首帧 + 同一串按键指令</b>下生成的。
  下方 HUD 跟着播放走：亮青的键是这一帧按下的指令，时间线上的竖线是当前位置。
  要看的是 —— 指令说往右，生成的画面有没有跟着往右。</p>
  <div class="meta" id="meta"></div>
  <div class="band" id="band"></div>
</header>
<div class="note">
  <b>只有 8 个按键是模型的输入。</b>下方时间线里橙蓝两色的 6 行相机增量
  （<code>d_pitch / d_yaw / d_roll / d_x / d_y / d_z</code>）<b>没有</b>喂给模型 ——
  它们是 GT 里实际发生的相机运动，画在这里当参照物：指令说 D（右移），
  GT 的 <code>d_x</code> 确实在动，那就可以直接比对生成结果有没有跟着动。
</div>
<div class="toolbar">
  <button id="allPlay">全部播放</button>
  <button id="allPause">全部暂停</button>
  <button id="allReset">回到开头</button>
  <span class="hint">时间线可点击跳转 · 两侧视频默认同步</span>
</div>
<main id="list"></main>
<footer id="foot"></footer>
</div>
<script>
const META = __META__, SAMPLES = __SAMPLES__, CAMCOLS = __CAMCOLS__;
const KEYPAD = [["移动",[[null,"W",null],["A","S","D"]]],["视角",[[null,"I",null],["J","K","L"]]]];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

document.getElementById("eyebrow").textContent =
  `${META.trainN} 条训练 · ${META.ckpt} · ${META.actionDim}D ${META.cols.join(" ")} · ${META.cond}`;
document.getElementById("meta").innerHTML = [
  [SAMPLES.length, "对比样本"], [META.actionDim + "D", "动作通道"],
  [META.steps, "采样步数"], [META.size, "分辨率"], [META.fps + " fps", "帧率"],
].map(([a, b]) => `<div><b>${a}</b><span>${b}</span></div>`).join("");
document.getElementById("band").innerHTML =
  Array.from({length: 14}, (_, i) =>
    `<i style="background:${i < 8 ? "var(--signal)" : (i < 11 ? "var(--warm)" : "var(--cool)")}"></i>`).join("");
document.getElementById("foot").textContent =
  `${META.run} · 生成于 ${META.created} · 页面由 code/abot/build_compare_page.py 产出`;

const ALL = [];
document.getElementById("allPlay").onclick = () => ALL.forEach(v => v.play());
document.getElementById("allPause").onclick = () => ALL.forEach(v => v.pause());
document.getElementById("allReset").onclick = () => ALL.forEach(v => v.currentTime = 0);

const list = document.getElementById("list");
SAMPLES.forEach((s, i) => {
  const el = document.createElement("section");
  el.className = "card";
  const totals = META.cols.map(k => `${k}:${s.keyTotals[k] || 0}`).join("  ");
  el.innerHTML = `
    <div class="card-head">
      <span class="idx">${String(i + 1).padStart(2, "0")}</span>
      <span class="sid">${s.short}</span>
      <span class="chip">seed ${s.seed}</span>
      <span class="chip">${s.frames} 帧</span>
      <span class="chip mode">${s.mode}</span>
      <span class="spacer"></span>
      <span class="frame-ind">帧 <b class="fnum">1</b>/${s.frames}</span>
    </div>
    <div class="stage">
      <div class="pane"><span class="pane-tag">GT</span>
        <video class="gt" playsinline loop muted preload="metadata" src="${s.gt}"></video></div>
      <div class="pane gen"><span class="pane-tag">GENERATED</span>
        <video class="gen" playsinline loop muted preload="metadata" src="${s.gen}"></video></div>
    </div>
    <div class="hud">
      <div class="keys"></div>
      <div class="tl"><canvas></canvas><div class="cursor" style="left:0"></div>
        <div class="tl-legend">
          <span><i class="sw" style="background:var(--signal)"></i>按键（模型输入）</span>
          <span><i class="sw" style="background:var(--warm)"></i>相机 正</span>
          <span><i class="sw" style="background:var(--cool)"></i>相机 负 — GT 参照，非输入</span>
        </div></div>
    </div>
    <div class="foot">
      <label><input type="checkbox" class="sync" checked> 同步播放</label>
      <label><input type="checkbox" class="play"> 播放</label>
    </div>
    <details><summary>prompt</summary><p>${s.prompt}</p></details>`;
  list.appendChild(el);
  wire(el, s, totals);
});

function wire(el, s, totals) {
  const gt = el.querySelector("video.gt"), gen = el.querySelector("video.gen");
  ALL.push(gt, gen);
  const kbox = el.querySelector(".keys"), cv = el.querySelector("canvas");
  const cur = el.querySelector(".cursor"), fnum = el.querySelector(".fnum");

  // 键盘：布局照搬 viz_action.py —— WASD 和 IJKL 分两组，一眼能分开"走"和"看"
  const kmap = {};
  KEYPAD.forEach(([title, rows]) => {
    const g = document.createElement("div");
    g.innerHTML = `<div class="klabel">${title}</div>`;
    rows.forEach(r => {
      const rd = document.createElement("div"); rd.className = "krow";
      r.forEach(k => {
        const d = document.createElement("div");
        d.className = "key" + (k === null ? " void" : "");
        d.textContent = k || "";
        if (k) { kmap[k] = d; if (!(s.keyTotals[k] > 0)) d.classList.add("void"); }
        rd.appendChild(d);
      });
      g.appendChild(rd);
    });
    kbox.appendChild(g);
  });

  const N = s.frames, KH = 13, CH = 11, PAD = 46;
  const H = KH * META.cols.length + CH * CAMCOLS.length + 8;
  function draw() {
    const W = cv.clientWidth || 900, dpr = devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    g.clearRect(0, 0, W, H);
    const cw = (W - PAD) / N;
    g.font = '9px "IBM Plex Mono",monospace'; g.textBaseline = "middle";
    let y = 0;
    META.cols.forEach(k => {
      const live = s.keyTotals[k] > 0;
      g.fillStyle = live ? css("--ink") : css("--muted");
      g.textAlign = "right"; g.fillText(k, PAD - 6, y + KH / 2);
      g.fillStyle = css("--signal");
      (s.keys[k] || []).forEach(([b, e]) => g.fillRect(PAD + b * cw, y + 1, (e - b) * cw, KH - 3));
      y += KH;
    });
    CAMCOLS.forEach(c => {
      g.fillStyle = css("--muted"); g.textAlign = "right";
      g.fillText(c.replace("d_", "").replace("_right", "x").replace("_down", "y").replace("_fwd", "z"),
                 PAD - 6, y + CH / 2);
      const v = s.cam[c] || [];
      for (let f = 0; f < N; f++) {
        const a = Math.abs(v[f] || 0) / 100; if (a < .03) continue;
        g.globalAlpha = a; g.fillStyle = (v[f] > 0) ? css("--warm") : css("--cool");
        g.fillRect(PAD + f * cw, y + 1, Math.max(cw, .7), CH - 3);
      }
      g.globalAlpha = 1; y += CH;
    });
  }
  draw();
  addEventListener("resize", draw);
  matchMedia("(prefers-color-scheme:dark)").addEventListener("change", draw);

  let f = -1;
  function tick() {
    const nf = Math.min(N - 1, Math.floor(gt.currentTime * META.fps));
    if (nf !== f) {
      f = nf; fnum.textContent = f + 1;
      const W = cv.clientWidth || 900;
      cur.style.left = (PAD + (f + .5) * (W - PAD) / N) + "px";
      META.cols.forEach(k => {
        const on = (s.keys[k] || []).some(([b, e]) => f >= b && f < e);
        kmap[k].classList.toggle("on", on);
      });
    }
    requestAnimationFrame(tick);
  }
  gt.addEventListener("loadedmetadata", () => requestAnimationFrame(tick), {once: true});

  const sync = el.querySelector(".sync"), play = el.querySelector(".play");
  // 两条视频必须同步才有可比性；用 GT 当主时钟，偏差超过 1 帧才纠正，
  // 免得每帧 seek 把播放卡成幻灯片。
  gt.addEventListener("timeupdate", () => {
    if (sync.checked && Math.abs(gen.currentTime - gt.currentTime) > 1 / META.fps) {
      gen.currentTime = gt.currentTime;
    }
  });
  play.addEventListener("change", () => {
    if (play.checked) { gt.play(); if (sync.checked) gen.play(); }
    else { gt.pause(); gen.pause(); }
  });
  cv.addEventListener("click", ev => {
    const r = cv.getBoundingClientRect();
    const t = Math.max(0, (ev.clientX - r.left - PAD) / (r.width - PAD)) * N / META.fps;
    gt.currentTime = t; if (sync.checked) gen.currentTime = t;
  });
}
</script></body></html>
"""

if __name__ == "__main__":
    main()
