#!/usr/bin/env python3
"""把 8 卡推理的结果和它用的逐 latent 动作文本渲成一页：生成 vs GT + 37 步标注同步高亮。

和 `build_action_viz.py`（标注核对台）的区别：那个页面核的是"标注跟 **GT** 对不对得上"，
这个核的是"**生成的画面**跟标注对不对得上"—— 前者验数据，后者验模型。

两个视频共用一个播放头，下方 37 格轨道按当前 latent 高亮，
当前那条标注在读数条里放大显示。判断依据就是：读着标注看生成画面，动作对不对得上。

同时产出两份：本地整页（`--out`）和 Artifact 片段（`--artifact-out`，
不带 doctype/html/head/body，由 Artifact 发布时套骨架）。两份共用同一套 STYLE/BODY。

用法:
    python3 code/abot/build_text_infer_viz.py \\
        --runs output/abot_inference/step9840_text_gpu* \\
        --out docs/action_prompt_text_viz.html \\
        --artifact-out .cache/artifact_text_viz.html
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abot_action as A  # noqa: E402
import action_script as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data/clips"
META = ROOT / "data/abot_meta_test_128.jsonl"
FFMPEG = (ROOT / "../envs/minimax_h3/lib/python3.10/site-packages/imageio_ffmpeg/"
          "binaries/ffmpeg-linux-x86_64-v7.0.2").resolve()
FPS = 24.156
LATENT_T = 37
NUM_FRAMES = 124


def camera_category(clause: str) -> str:
    """相机从句归到六类之一 —— 时间轴的颜色和图例都用它。判定与
    `build_action_viz.py` 保持一致，两个核对台的配色才对得上。"""
    if clause == S.CAMERA_IDLE:
        return "static"
    if clause == S.CAMERA_FOLLOW:
        return "follow"
    if "drifts" in clause:
        return "drift"
    if "pans" in clause and "tilts" in clause:
        return "pantilt"
    if "pans" in clause:
        return "pan"
    return "tilt"


def encode_video(src: Path, crf: int, width: int) -> str:
    """转成小体积 mp4 再内联。16 条原视频合计 ~150 MB，直接嵌会做出个打不开的页面。"""
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "v.mp4"
        subprocess.run(
            [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(src),
             "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-an", str(dst)],
            check=True)
        return "data:video/mp4;base64," + base64.b64encode(dst.read_bytes()).decode()


def build_sample(run_dir: Path, rows: dict, crf: int, width: int) -> dict | None:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return None
    man = json.loads(manifest.read_text())
    entries = [s for s in man["samples"] if s.get("status") == "complete"]
    if not entries:
        return None
    ent = entries[0]
    sid = ent["sample_id"]
    row = rows[sid]

    sample_dir = run_dir / f"{sid}_w{ent['window']:03d}"
    gen, gt = sample_dir / "generated.mp4", sample_dir / "gt.mp4"
    if not gen.is_file():
        return None

    # 动作文本是**纯函数**：同一段按键永远得到同一串文本，
    # 所以这里重算一遍与推理时喂进去的完全一致，不必从缓存里捞。
    pooled = A.bin_to_latent(np.load(CLIPS / row["action"])[:NUM_FRAMES], LATENT_T)
    k9 = S.keys9(pooled)
    script = S.annotate_from_keys9(k9)
    rate = S._camera_rate(pooled)

    spans = [e - s for s, e in A.frame_spans(LATENT_T)]
    starts = np.concatenate([[0], np.cumsum(spans)[:-1]])
    steps = []
    for k in range(LATENT_T):
        motion, camera = script[k][len("the man "):].split(", camera ")
        steps.append({
            "t": round(float(starts[k] / FPS), 3),
            "n": int(spans[k]),
            "k9": [int(x) for x in k9[k]],
            "yaw": round(float(rate[k, 1]), 3),
            "motion": motion,
            "camera": camera,
            "text": script[k],
            "cat": camera_category(camera),
        })

    print(f"  {sid[:8]}  编码视频 …", flush=True)
    return {
        "id": sid[:8],
        "full": sid,
        "gpu": run_dir.name.rsplit("_", 1)[-1],
        "scene": row["prompt"],
        "gen": encode_video(gen, crf, width),
        "gt": encode_video(gt, crf, width) if gt.is_file() else None,
        "steps": steps,
        "distinct": len(set(script)),
        "active": [K for i, K in enumerate(S.KEYS9) if k9[:, i].any()],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/action_prompt_text_viz.html")
    ap.add_argument("--artifact-out", type=Path, default=None)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--width", type=int, default=624)
    args = ap.parse_args()

    rows = {json.loads(l)["sample_id"]: json.loads(l)
            for l in open(META) if l.strip()}

    samples, ckpt, cfg = [], None, None
    for run in sorted(args.runs):
        s = build_sample(run, rows, args.crf, args.width)
        if s is None:
            print(f"  跳过 {run.name}（没有完成的样本）", flush=True)
            continue
        man = json.loads((run / "manifest.json").read_text())
        ckpt = ckpt or man["checkpoint"]
        cfg = cfg or man["config"]
        samples.append(s)
    if not samples:
        raise SystemExit("没有可用的推理结果")

    vocab = Counter(st["text"] for s in samples for st in s["steps"])
    meta = {
        "checkpoint": Path(ckpt["path"] if isinstance(ckpt, dict) else ckpt).name,
        "steps": cfg.get("steps"),
        "cfg_scale": cfg.get("cfg_scale", 1.0),
        "seed": cfg.get("generation_seed"),
        "n": len(samples),
        "vocab": vocab.most_common(),
        "cat": {st["text"]: st["cat"] for s_ in samples for st in s_["steps"]},
        "keys9": S.KEYS9,
    }
    body = (BODY
            .replace("__SAMPLES__", json.dumps(samples, ensure_ascii=False))
            .replace("__META__", json.dumps(meta, ensure_ascii=False)))

    for path, text in [
        (args.out, PAGE.replace("__HEAD__", TITLE + FONTS + STYLE).replace("__BODY__", body)),
        (args.artifact_out, TITLE + FONTS + STYLE + body),
    ]:
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"\n写出 {path}  ({path.stat().st_size / 2**20:.1f} MB, "
              f"{len(samples)} 个样本)")


TITLE = "<title>推理结果核对台</title>\n"

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

# 版式沿用 build_action_viz.py 的标注核对台 —— 同一套配色、字号、九宫格按键 HUD、
# 按相机类别着色的时间轴。差别只在 stage 里放两个视频：左 GT、右推理结果。
STYLE = r"""<style>
:root{
  --ground:#EEF1F3; --surface:#FFF; --sunken:#F6F8F9; --line:#D3DAE0;
  --ink:#10151C; --body:#3A444F; --muted:#6E7985;
  --accent:#17A791;
  --c-pan:#17A791; --c-pantilt:#3B72C4; --c-tilt:#7A5AA8;
  --c-follow:#C56A1E; --c-drift:#8A7A1E; --c-static:#9AA3AD;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0B0E13; --surface:#151B24; --sunken:#0F141B; --line:#27313D;
  --ink:#EAEEF2; --body:#B3BDC8; --muted:#79838F;
  --accent:#3ECFB6;
  --c-pan:#3ECFB6; --c-pantilt:#5590E0; --c-tilt:#A98CD8;
  --c-follow:#E68A3C; --c-drift:#C4B44A; --c-static:#5C6672;
}}
:root[data-theme="dark"]{
  --ground:#0B0E13; --surface:#151B24; --sunken:#0F141B; --line:#27313D;
  --ink:#EAEEF2; --body:#B3BDC8; --muted:#79838F;
  --accent:#3ECFB6;
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
.lede{font-size:17px;max-width:68ch;margin:0 0 22px}
.lede b{color:var(--ink);font-weight:600}
h2{font-size:22px;font-weight:600;color:var(--ink);margin:48px 0 8px;letter-spacing:-.01em}
p{max-width:70ch}
code{font-family:var(--mono);font-size:.87em;background:var(--sunken);border:1px solid var(--line);
     border-radius:3px;padding:1px 5px;color:var(--ink)}
.sub{color:var(--muted);max-width:72ch;margin:0 0 18px}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;
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
.idx{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--accent)}
.sid{font-family:var(--mono);font-size:13px;color:var(--ink)}
.chip{font-family:var(--mono);font-size:11px;color:var(--muted);border:1px solid var(--line);
      border-radius:3px;padding:1px 7px}
.spacer{flex:1}

/* stage 比原版宽：这里要并排放两个视频。左 GT、右推理结果。 */
.stage{max-width:1180px;margin:0 auto;padding:20px 18px 4px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.pair{grid-template-columns:1fr}}
.pane{min-width:0}
.vhead{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:10.5px;
       letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
.vhead i{display:inline-block;width:8px;height:8px;border-radius:2px}
.vhead.gt i{background:var(--muted)} .vhead.gen i{background:var(--accent)}
.vhead.gen{color:var(--accent)}
video{width:100%;display:block;border-radius:7px;background:#000}
.strip{display:flex;gap:1px;height:22px;margin-top:12px;border-radius:4px;overflow:hidden;
       border:1px solid var(--line)}
.strip div{cursor:pointer;transition:filter .12s}
.strip div:hover{filter:brightness(1.3)}
.strip div.on{outline:2px solid var(--ink);outline-offset:-2px;z-index:2}
.axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;
      color:var(--muted);margin-top:4px}

.now{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin:14px 0 4px;
     padding:14px 18px;background:var(--sunken);border:1px solid var(--line);border-radius:7px;
     min-height:64px}
.now .txt{font-size:17px;line-height:1.4;color:var(--body);flex:1;min-width:260px}
.now .txt b{color:var(--ink);font-weight:600}
.now .k{font-family:var(--mono);font-size:11px;color:var(--muted);white-space:nowrap}

/* 与 raw data HUD (viz_action.py draw_keys) 同构：两个九宫格，移动与视角分开，
   按下 = 实心填充。大号用于「当前标注」读数，小号用于滚栏，小号隐去字母只留方位。 */
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
.pad.sm{gap:2px}
.pad.sm .lbl{display:none}
.pad.sm .r{gap:2px}
.pad.sm .k{width:12px;height:12px;border-width:1.5px;border-radius:3px;font-size:0}
.pad.sm .k.wide{width:40px;height:9px}

.rows{padding:6px 0 12px;max-height:264px;overflow-y:auto;border-top:1px solid var(--line)}
.row{display:grid;grid-template-columns:66px 176px 62px 1fr;gap:12px;align-items:center;
     padding:5px 22px;font-size:13.5px;border-left:3px solid transparent;cursor:pointer}
.row:hover{background:var(--sunken)}
.row.on{background:var(--sunken);border-left-color:var(--ink)}
.row .k{font-family:var(--mono);font-size:11px;color:var(--muted);
        font-variant-numeric:tabular-nums}
.row .num{font-family:var(--mono);font-size:11px;color:var(--muted);
          font-variant-numeric:tabular-nums;text-align:right}
.row .txt{color:var(--body)}
.row .txt b{color:var(--ink);font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:7px;
     vertical-align:baseline}

.scroll{overflow-x:auto;margin:18px 0}
table{width:100%;border-collapse:collapse;font-size:14px;min-width:600px;
      background:var(--surface);border:1px solid var(--line);border-radius:7px}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-family:var(--mono);font-size:10.5px;font-weight:500;letter-spacing:.08em;
         text-transform:uppercase;color:var(--muted);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
td.nm{font-family:var(--mono);font-size:12.5px;color:var(--ink)}
td.num{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums;
       text-align:right;color:var(--ink);white-space:nowrap}
.bar{display:inline-block;height:8px;border-radius:2px;background:var(--accent);opacity:.75;
     vertical-align:middle;margin-left:8px}
.note{background:var(--sunken);border:1px solid var(--line);border-left:3px solid var(--accent);
      border-radius:0 7px 7px 0;padding:14px 18px;margin:20px 0}
.note p{margin:0;max-width:none}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
"""

BODY = r"""<div class="wrap">
<header>
  <p class="eyebrow">MiniMax-H3 · ABot 逐 latent 文本注入</p>
  <h1>推理结果核对台</h1>
  <p class="lede">每个 latent 一条结构化文本条件，是 <b>9 位按键的纯函数</b>，
  经文本通路喂进模型 —— 条件通路<b>零新增可训练参数</b>。
  左边是<b>真实片段</b>，右边是<b>模型生成的</b>，两边共用一个播放头。
  这页拿来<b>肉眼核对模型有没有听按键的话</b>：读着当前那条标注，看右边动作对不对得上，
  再跟左边比。权重量级这类统计量验得了「在学」，验不了这个。</p>
  <div class="stats" id="stats"></div>
  <div class="legend" id="legend"></div>
</header>

<div id="clips"></div>

<h2>这批样本用到的动作文本</h2>
<p class="sub">模板固定，只换两个槽：<code>the man &lt;怎么动&gt;, camera &lt;相机视角怎么动&gt;</code>。
两个槽都是这 9 位的<b>确定性函数</b> —— 同样的按键永远得到同样的串，
所以这页里重算的文本与推理时喂进去的逐字一致。</p>
<div class="scroll"><table id="tvocab"></table></div>

<div class="note">
<p><b>这页能判什么、不能判什么。</b>肉眼看八条样本能看出动作大体听不听话，
但看不出「绑定在统计上成立」。真正的判据是<b>只改第 k 条标注、其余全不动，
测输出的哪些帧变了</b> —— 变化集中在第 k 帧及邻域才算绑定生效。那个工具还没写。
注意 37×37 注意力的对角占优在硬掩码下是<b>架构保证</b>的、恒为真，不能拿来当判据。</p>
</div>
</div>
<script>
// 所有 const 必须声明在使用它们的代码之前：函数声明会提升、const 不会，
// 顺序错了整个 script 抛 ReferenceError，页面一片空白，而静态检查全是绿的。
const CLIPS = __SAMPLES__, META = __META__;
const CATNAME = {pan:"摇镜", pantilt:"摇镜+俯仰", tilt:"俯仰",
                 follow:"跟随", drift:"平移", static:"静止"};
const CATORDER = ["pan","pantilt","tilt","follow","drift","static"];
const KEYS9 = META.keys9;
const cv = c => `var(--c-${c})`;
const esc = s => String(s).replace(/[&<>]/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[m]));

const KIDX = Object.fromEntries(KEYS9.map((n, i) => [n, i]));
// 键盘上的物理排布，与 viz_action.py 的 KEYPAD 一致
const PADS = [["移动", [[null,"W",null],["A","S","D"]]], ["视角", [[null,"I",null],["J","K","L"]]]];
function padsHtml(k9, sm) {
  const cls = sm ? "pad sm" : "pad";
  const cell = n => n === null
    ? `<div class="k hole"></div>`
    : `<div class="k ${k9[KIDX[n]] ? "on" : ""}">${sm ? "" : n}</div>`;
  const pads = PADS.map(([lbl, rows]) => `<div class="${cls}">
      <span class="lbl">${lbl}</span>
      ${rows.map(r => `<div class="r">${r.map(cell).join("")}</div>`).join("")}
    </div>`);
  // F 单独一格：它不是原始录像里的键，是从 COLMAP 速率合成的第 9 位
  pads.push(`<div class="${cls}">
      <span class="lbl">快</span>
      <div class="r"><div class="k ${sm ? "" : "wide "}${k9[KIDX.F] ? "on" : ""}">${sm ? "" : "FAST"}</div></div>
    </div>`);
  return `<div class="pads">${pads.join("")}</div>`;
}

document.getElementById("stats").innerHTML = [
  [META.n, "条样本 · 8 卡并行"],
  [META.checkpoint.replace(/\.safetensors$/, ""), "checkpoint"],
  [META.steps, "去噪步数"],
  [META.cfg_scale, "cfg_scale"],
  [META.vocab.length, "种文本（本批去重）"],
].map(([b, s]) => `<div class="stat"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");

document.getElementById("legend").innerHTML = CATORDER
  .map(c => `<span><i style="background:${cv(c)}"></i>${CATNAME[c]}</span>`).join("");

document.getElementById("clips").innerHTML = CLIPS.map((c, ci) => {
  const total = c.steps.reduce((a, s) => a + s.n, 0);
  const strip = c.steps.map((s, k) =>
    `<div data-c="${ci}" data-k="${k}" title="步 ${k} · ${esc(s.text)}"
          style="flex:${s.n};background:${cv(s.cat)}"></div>`).join("");
  const rows = c.steps.map((s, k) => `<div class="row" data-c="${ci}" data-k="${k}">
      <span class="k">${k} · ${s.t.toFixed(2)}s</span>
      ${padsHtml(s.k9, true)}
      <span class="num">${s.yaw >= 0 ? "+" : ""}${s.yaw.toFixed(3)}</span>
      <span class="txt"><i class="dot" style="background:${cv(s.cat)}"></i>the man
        <b>${esc(s.motion)}</b>, camera <b>${esc(s.camera)}</b></span>
    </div>`).join("");
  return `<div class="card">
    <div class="card-head">
      <span class="idx">${String(ci + 1).padStart(2, "0")}</span>
      <span class="sid">${esc(c.id)}</span>
      <span class="chip">${esc(c.gpu)}</span>
      <span class="chip">${c.distinct} 种 / 37 步</span>
      <span class="spacer"></span>
      <span class="chip">按下过 ${esc(c.active.join(" ") || "无")}</span>
      <span class="chip">124 帧 · ${(total / 24.156).toFixed(2)}s</span>
    </div>
    <div class="stage">
      <div class="pair">
        <div class="pane"><p class="vhead gt"><i></i>真实片段 GT</p>
          ${c.gt ? `<video src="${c.gt}" controls loop muted playsinline
                      data-c="${ci}" data-role="gt"></video>`
                 : '<p class="sub">无 GT</p>'}</div>
        <div class="pane"><p class="vhead gen"><i></i>模型生成</p>
          <video src="${c.gen}" controls loop muted playsinline
                 data-c="${ci}" data-role="gen"></video></div>
      </div>
      <div class="strip" data-c="${ci}">${strip}</div>
      <div class="axis"><span>0.00s</span><span>latent 0 → 36</span>
        <span>${(total / 24.156).toFixed(2)}s</span></div>
      <div class="now" id="now-${ci}"></div>
    </div>
    <div class="rows" data-c="${ci}">${rows}</div>
  </div>`;
}).join("");

// 播放时同步高亮当前 latent
function highlight(ci, k) {
  const st = CLIPS[ci].steps[k];
  const now = document.getElementById(`now-${ci}`);
  if (now) now.innerHTML =
    `${padsHtml(st.k9)}
     <span class="txt"><i class="dot" style="background:${cv(st.cat)}"></i>the man
       <b>${esc(st.motion)}</b>, camera <b>${esc(st.camera)}</b></span>
     <span class="k">latent ${k} · ${st.t.toFixed(2)}s · d_yaw ${st.yaw >= 0 ? "+" : ""}${st.yaw.toFixed(3)}</span>`;
  document.querySelectorAll(`.strip[data-c="${ci}"] > div`).forEach((d, i) =>
    d.classList.toggle("on", i === k));
  const rows = document.querySelectorAll(`.rows[data-c="${ci}"] .row`);
  rows.forEach((r, i) => r.classList.toggle("on", i === k));
  const row = rows[k], box = document.querySelector(`.rows[data-c="${ci}"]`);
  if (row && box) {
    const top = row.offsetTop - box.offsetTop, h = box.clientHeight;
    if (top < box.scrollTop || top > box.scrollTop + h - row.clientHeight - 8)
      box.scrollTo({top: top - h / 2, behavior: "smooth"});
  }
}
function stepAt(ci, t) {
  const st = CLIPS[ci].steps;
  let k = 0;
  while (k + 1 < st.length && st[k + 1].t <= t) k++;
  return k;
}
CLIPS.forEach((_, ci) => highlight(ci, 0));

// 一张卡里的两个视频互相跟随。阈值判断是必需的：无条件对拷 currentTime
// 会让两边的 seeked 事件互相触发，无限乒乓。
function pairOf(v) {
  return document.querySelector(
    `video[data-c="${v.dataset.c}"][data-role="${v.dataset.role === "gt" ? "gen" : "gt"}"]`);
}
document.querySelectorAll("video").forEach(v => {
  const ci = +v.dataset.c;
  v.addEventListener("timeupdate", () => highlight(ci, stepAt(ci, v.currentTime)));
  v.addEventListener("play", () => {
    const o = pairOf(v);
    if (!o) return;
    if (Math.abs(o.currentTime - v.currentTime) > 0.15) o.currentTime = v.currentTime;
    o.play().catch(() => {});
  });
  v.addEventListener("pause", () => { const o = pairOf(v); if (o) o.pause(); });
  v.addEventListener("seeked", () => {
    const o = pairOf(v);
    if (o && Math.abs(o.currentTime - v.currentTime) > 0.15) o.currentTime = v.currentTime;
  });
});
// 点时间轴或某一行都能跳到那一步，两个视频一起跳
document.addEventListener("click", e => {
  const el = e.target.closest("[data-k]");
  if (!el) return;
  const ci = +el.dataset.c, k = +el.dataset.k, t = CLIPS[ci].steps[k].t;
  document.querySelectorAll(`video[data-c="${ci}"]`).forEach(v => { v.currentTime = t; });
  highlight(ci, k);
});

// 页尾：本批文本词表
const pct = x => (x * 100).toFixed(1) + "%";
const totalLines = META.vocab.reduce((a, b) => a + b[1], 0);
document.getElementById("tvocab").innerHTML =
  `<thead><tr><th>类别</th><th>动作文本</th><th style="text-align:right">次数</th>
    <th style="text-align:right">占比</th><th></th></tr></thead><tbody>` +
  META.vocab.map(([t, n]) => {
    const c = META.cat[t] || "static";
    return `<tr><td class="nm"><i class="dot" style="background:${cv(c)}"></i>${CATNAME[c]}</td>
      <td class="nm" style="color:var(--body)">${esc(t)}</td>
      <td class="num">${n}</td><td class="num">${pct(n / totalLines)}</td>
      <td><span class="bar" style="width:${Math.max(2, n / totalLines * 900).toFixed(0)}px;
        background:${cv(c)}"></span></td></tr>`;
  }).join("") + `</tbody>`;
</script>
"""


if __name__ == "__main__":
    main()
