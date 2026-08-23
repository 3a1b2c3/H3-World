#!/usr/bin/env python3
"""同首帧换 action 的对照台：任选两个按键预设并排比，外加全部 8 个的位移测量。

与另外两个核对台的分工：
  * `build_action_viz.py`      标注 vs **GT**        —— 验数据
  * `build_text_infer_viz.py`  生成 vs GT            —— 验模型跟不跟得上真实动作
  * 本页                        生成 vs **另一个生成** —— 验换了 action 出不出不同且符合的结果

首帧 / 场景描述 / seed 全部相同，唯一变量是按键，所以两段视频的任何差异都归因于
动作文本。8 段视频只内联一次，两个播放位从同一份数据里挑，页面不会因为并排而翻倍。

位移用相位相关估（与 `analyze_probe.py` 同一套，符号已在 128 条测试集上标定：
cum_dx 为负 = 相机左摇，r = −0.936、14/14 反号）。

用法:
    python3 code/abot/build_action_ab_viz.py --tag ab8_step9840 \\
        --out docs/action_ab_viz.html --artifact-out .cache/artifact_ab_viz.html
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
import action_script as S  # noqa: E402
from analyze_probe import track  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INFER = ROOT / "output/abot_inference"
FFMPEG = (ROOT / "../envs/minimax_h3/lib/python3.10/site-packages/imageio_ffmpeg/"
          "binaries/ffmpeg-linux-x86_64-v7.0.2").resolve()
LATENT_T = 37

# 与 infer_abot.ACTION_PRESETS 对应，外加一句人话说明这个预设在测什么
PRESET_NOTE = {
    "still":          "基线：不按任何键",
    "forward":        "只按 W",
    "back":           "只按 S",
    "strafe-left":    "只按 A",
    "strafe-right":   "只按 D",
    "pan-left":       "只按 J",
    "pan-right":      "只按 L",
    "pan-right-fast": "按 L + 速度档 F",
}
ORDER = list(PRESET_NOTE)


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


def build(tag: str, crf: int, width: int) -> tuple[list[dict], dict]:
    import infer_abot as I

    items, sid, ckpt, cfg = [], None, None, None
    for name in ORDER:
        run = INFER / f"{tag}_{name}"
        gen = next(run.glob("*/generated.mp4"), None)
        if gen is None:
            print(f"  跳过 {name}（没有产物）", flush=True)
            continue
        man = json.loads((run / "manifest.json").read_text())
        ent = man["samples"][0]
        sid = sid or ent["sample_id"]
        ckpt = ckpt or man["checkpoint"]
        cfg = cfg or man["config"]

        k9 = I.preset_keys9(name)
        script = S.annotate_from_keys9(k9)
        assert len(set(script)) == 1, f"{name}: 预设应当全程同一句，实得 {len(set(script))} 句"
        print(f"  {name:<16} 测位移 + 编码 …", flush=True)
        items.append({
            "name": name,
            "note": PRESET_NOTE[name],
            "keys": [K for i, K in enumerate(S.KEYS9) if k9[0, i]],
            "k9": [int(x) for x in k9[0]],
            "text": script[0],
            "cum_dx": round(float(track(gen)["cum_dx"]), 1),
            "video": encode_video(gen, crf, width),
        })

    base = next((x["cum_dx"] for x in items if x["name"] == "still"), 0.0)
    for x in items:
        x["rel"] = round(x["cum_dx"] - base, 1)

    row = next(iter(json.loads(l) for l in open(ROOT / "data/abot_meta_test_128.jsonl")
                    if json.loads(l)["sample_id"] == sid))
    meta = {
        "sample": sid,
        "scene": row["prompt"],
        "checkpoint": Path(ckpt["path"] if isinstance(ckpt, dict) else ckpt).name,
        "steps": cfg.get("steps"),
        "cfg_scale": cfg.get("cfg_scale", 1.0),
        "seed": cfg.get("generation_seed"),
        "keys9": S.KEYS9,
        "n": len(items),
    }
    return items, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ab8_step9840")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/action_ab_viz.html")
    ap.add_argument("--artifact-out", type=Path, default=None)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--width", type=int, default=624)
    args = ap.parse_args()

    items, meta = build(args.tag, args.crf, args.width)
    if not items:
        raise SystemExit("没有可用的对照结果")
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
        print(f"\n写出 {path}  ({path.stat().st_size / 2**20:.1f} MB, {len(items)} 个预设)")


TITLE = "<title>换按键对照台</title>\n"

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

# 版式沿用标注核对台。新增的只有发散配色的位移图：
# 左/右是**极性**，所以两个色极 + 中性灰中点，不是分类色。
# 两极经 validate_palette.js 六项全过（亮 ΔE 28.3 / 暗 ΔE 26.6，色盲分离充裕）。
STYLE = r"""<style>
:root{
  --ground:#EEF1F3; --surface:#FFF; --sunken:#F6F8F9; --line:#D3DAE0;
  --ink:#10151C; --body:#3A444F; --muted:#6E7985;
  --accent:#17A791;
  --neg:#3B72C4; --pos:#C56A1E; --zero:#9AA3AD;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0B0E13; --surface:#151B24; --sunken:#0F141B; --line:#27313D;
  --ink:#EAEEF2; --body:#B3BDC8; --muted:#79838F;
  --accent:#3ECFB6;
  --neg:#5590E0; --pos:#C8752B; --zero:#5C6672;
}}
:root[data-theme="dark"]{
  --ground:#0B0E13; --surface:#151B24; --sunken:#0F141B; --line:#27313D;
  --ink:#EAEEF2; --body:#B3BDC8; --muted:#79838F;
  --accent:#3ECFB6;
  --neg:#5590E0; --pos:#C8752B; --zero:#5C6672;
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
.sub{color:var(--muted);max-width:74ch;margin:0 0 18px}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;
       background:var(--line);border:1px solid var(--line);border-radius:7px;
       overflow:hidden;margin:0 0 10px}
.stat{background:var(--surface);padding:14px 17px}
.stat b{display:block;font-family:var(--mono);font-size:20px;color:var(--ink);
        font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stat span{display:block;font-family:var(--mono);font-size:10px;letter-spacing:.08em;
           text-transform:uppercase;color:var(--muted);margin-top:3px}

.card{background:var(--surface);border:1px solid var(--line);border-radius:9px;
      margin:22px 0;overflow:hidden}
.card-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 18px;
           border-bottom:1px solid var(--line)}
.sid{font-family:var(--mono);font-size:13px;color:var(--ink)}
.chip{font-family:var(--mono);font-size:11px;color:var(--muted);border:1px solid var(--line);
      border-radius:3px;padding:1px 7px}
.spacer{flex:1}
.scene{font-size:12.5px;color:var(--muted);margin:0;padding:12px 18px;
       border-bottom:1px solid var(--line);max-height:3.2em;overflow:hidden;cursor:pointer}
.scene.open{max-height:none}

.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:18px}
@media(max-width:860px){.pair{grid-template-columns:1fr}}
.pane{min-width:0;display:flex;flex-direction:column;gap:9px}
.vhead{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:10.5px;
       letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.vhead i{display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--muted)}
.vhead.gen{color:var(--accent)} .vhead.gen i{background:var(--accent)}
.idx{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--accent)}
select:disabled{opacity:.72;cursor:default}
select{font-family:var(--mono);font-size:13px;padding:7px 9px;border-radius:6px;
       border:1px solid var(--line);background:var(--sunken);color:var(--ink);width:100%}
select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
video{width:100%;display:block;border-radius:7px;background:#000}
.said{background:var(--sunken);border:1px solid var(--line);border-radius:7px;
      padding:11px 13px;min-height:96px}
.said .t{font-size:15.5px;color:var(--ink);line-height:1.4;margin-bottom:9px}
.said .m{font-family:var(--mono);font-size:11.5px;color:var(--muted);
         font-variant-numeric:tabular-nums}
.said .m b{color:var(--ink);font-weight:600}

/* 九宫格按键 HUD：与 viz_action.py 的 raw data HUD 同构 */
.pads{display:flex;gap:12px;align-items:flex-end;margin-top:9px}
.pad{display:flex;flex-direction:column;gap:4px}
.pad .lbl{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;color:var(--muted);
          text-transform:uppercase}
.pad .r{display:flex;gap:4px}
.pad .k{width:24px;height:24px;border-radius:5px;border:2px solid var(--line);
        font-family:var(--mono);font-size:11px;font-weight:600;color:var(--muted);
        display:flex;align-items:center;justify-content:center}
.pad .k.on{background:var(--accent);border-color:var(--accent);color:#fff}
.pad .k.hole{border-color:transparent}
.pad .k.wide{width:80px;height:18px;font-size:9.5px;border-radius:4px}

/* 发散条形图：零在中间，左负右正。轴线是中性灰，不参与语义。 */
.chart{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:20px}
.crow{display:grid;grid-template-columns:132px 1fr 74px;gap:12px;align-items:center;
      padding:5px 0;cursor:pointer;border-radius:5px}
.crow:hover{background:var(--sunken)}
.crow .nm{font-family:var(--mono);font-size:12px;color:var(--ink);white-space:nowrap}
.crow .val{font-family:var(--mono);font-size:12.5px;color:var(--ink);text-align:right;
           font-variant-numeric:tabular-nums}
.gut{position:relative;height:20px}
.gut .axis0{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--zero)}
.gut .b{position:absolute;top:3px;height:14px;border-radius:4px}
.gut .b.n{background:var(--neg)} .gut .b.p{background:var(--pos)}
.clegend{display:flex;gap:18px;font-family:var(--mono);font-size:11.5px;color:var(--muted);
         margin:2px 0 14px}
.clegend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px}
.cscale{display:grid;grid-template-columns:132px 1fr 74px;gap:12px;
        font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:6px}
.cscale .ticks{display:flex;justify-content:space-between}

.scroll{overflow-x:auto;margin:18px 0}
table{width:100%;border-collapse:collapse;font-size:14px;min-width:600px;
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
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
"""

BODY = r"""<div class="wrap">
<header>
  <p class="eyebrow">MiniMax-H3 · ABot 逐 latent 文本注入</p>
  <h1>换按键对照台</h1>
  <p class="lede">同一个首帧、同一个 seed、同一段场景描述，<b>唯一变量是按键</b>。
  所以两段视频之间的任何差异，都只能归因于逐 latent 的动作文本。
  八个预设一路滚下去，每张卡左边是对照（默认不按键的基线，可换）、右边是该预设；页尾是全部 8 个的位移测量 —— 判方向不靠肉眼。</p>
  <div class="stats" id="stats"></div>
</header>

<div class="card" id="head-card">
  <div class="card-head">
    <span class="sid" id="sid"></span>
    <span class="chip">同首帧 · 同 seed · 同场景描述</span>
    <span class="spacer"></span>
    <span class="chip" id="ckpt"></span>
  </div>
  <p class="scene" id="scene" title="点击展开完整场景描述"></p>
</div>

<div id="cards"></div>

<h2>累计水平位移</h2>
<p class="sub">相位相关逐帧估全局位移再累加。符号已在 128 条测试集上标定：
与 npy 的 Σd_yaw 相关系数 <code>r = −0.936</code>、14/14 反号，所以
<b>cum_dx 为负 = 相机左摇</b>。下面画的是<b>相对 <code>still</code> 基线</b>的偏移 ——
片段本身有固有运动，基线不是零。</p>
<div class="chart">
  <div class="clegend">
    <span><i style="background:var(--neg)"></i>向左（cum_dx 更负）</span>
    <span><i style="background:var(--pos)"></i>向右（cum_dx 更正）</span>
    <span><i style="background:var(--zero)"></i>零基线</span>
  </div>
  <div id="chart"></div>
  <div class="cscale"><span></span><div class="ticks" id="ticks"></div><span></span></div>
</div>

<h2>逐项数值</h2>
<div class="scroll"><table id="tbl"></table></div>

<div class="note">
<p><b>这页能判什么、不能判什么。</b>它能回答"换 action 出不出不同且符合的结果"——
方向、单调性、速度档都能量化。它<b>回答不了</b>"第 k 条标注绑不绑第 k 帧"：
整段按住同一组键时，就算模型只是笼统地响应了全段文本、完全没有逐帧绑定，
这些位移数字也会长一样。真判据是<b>只改第 k 条标注、其余全不动，测输出的哪些帧变了</b>。
另外这只是 <b>1 个首帧、1 个 seed</b>，是存在性证据，不是统计结论。</p>
</div>
</div>
<script>
// 所有 const 必须声明在使用它们的代码之前：函数声明会提升、const 不会，
// 顺序错了整个 script 抛 ReferenceError，页面一片空白，而静态检查全是绿的。
const ITEMS = __ITEMS__, META = __META__;
const KEYS9 = META.keys9;
const KIDX = Object.fromEntries(KEYS9.map((n, i) => [n, i]));
const PADS = [["移动", [[null,"W",null],["A","S","D"]]], ["视角", [[null,"I",null],["J","K","L"]]]];
const esc = s => String(s).replace(/[&<>"]/g, m =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[m]));

function padsHtml(k9){
  const cell = n => n === null
    ? '<div class="k hole"></div>'
    : `<div class="k ${k9[KIDX[n]] ? "on" : ""}">${n}</div>`;
  const pads = PADS.map(([lbl, rows]) => `<div class="pad"><span class="lbl">${lbl}</span>
      ${rows.map(r => `<div class="r">${r.map(cell).join("")}</div>`).join("")}</div>`);
  pads.push(`<div class="pad"><span class="lbl">快</span>
      <div class="r"><div class="k wide ${k9[KIDX.F] ? "on" : ""}">FAST</div></div></div>`);
  return `<div class="pads">${pads.join("")}</div>`;
}

function renderStats(){
  const sorted = [...ITEMS].sort((a, b) => a.rel - b.rel);
  document.getElementById("stats").innerHTML = [
    [META.n, "个按键预设"],
    [META.steps, "去噪步数"],
    [META.cfg_scale, "cfg_scale"],
    [META.seed, "seed（全部相同）"],
    [`${sorted[0].rel} … ${sorted[sorted.length - 1].rel}`, "相对基线位移跨度"],
  ].map(([b, s]) => `<div class="stat"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");
  document.getElementById("sid").textContent = META.sample.slice(0, 8);
  document.getElementById("ckpt").textContent = META.checkpoint;
  const sc = document.getElementById("scene");
  sc.textContent = META.scene;
  sc.onclick = () => sc.classList.toggle("open");
}

function paneHtml(cls, label){
  return `<div class="pane">
    <div class="vhead ${cls}"><i></i>${label}</div>
    <select></select>
    <video controls loop muted playsinline preload="metadata"></video>
    <div class="said"><div class="t"></div><div class="m"></div></div>
  </div>`;
}

// 一个预设一张卡，直接滚着看完 8 个 —— 不用先在下拉里挑。
// 左边是对照（默认 still 基线，可换成任意其它预设），右边固定是这张卡的预设。
function renderCards(){
  const opts = ITEMS.map((it, i) =>
    `<option value="${i}">${esc(it.name)} — ${esc(it.note)}</option>`).join("");
  const baseIdx = Math.max(0, ITEMS.findIndex(x => x.name === "still"));
  const host = document.getElementById("cards");

  host.innerHTML = ITEMS.map((it, ci) => {
    const sign = it.rel > 0 ? "+" : "";
    return `<div class="card" data-ci="${ci}">
      <div class="card-head">
        <span class="idx">${String(ci + 1).padStart(2, "0")}</span>
        <span class="sid">${esc(it.name)}</span>
        <span class="chip">${esc(it.note)}</span>
        <span class="spacer"></span>
        <span class="chip">cum_dx ${it.cum_dx}</span>
        <span class="chip">相对基线 ${sign}${it.rel}</span>
      </div>
      <div class="pair">${paneHtml("ref", "对照")}${paneHtml("gen", "该预设")}</div>
    </div>`;
  }).join("");

  Array.from(host.children).forEach((card, ci) => {
    const panes = card.querySelectorAll(".pane");
    const wire = (pane, idx, locked) => {
      const sel = pane.querySelector("select");
      const vid = pane.querySelector("video");
      const t = pane.querySelector(".t");
      const m = pane.querySelector(".m");
      sel.innerHTML = opts;
      sel.value = String(idx);
      if (locked) sel.disabled = true;
      function apply(){
        const it = ITEMS[+sel.value];
        vid.src = it.video;
        t.textContent = it.text;
        const sign = it.rel > 0 ? "+" : "";
        m.innerHTML = `按键 <b>${esc(it.keys.join(" ") || "无")}</b> &nbsp;·&nbsp; ` +
          `cum_dx <b>${it.cum_dx}</b> &nbsp;·&nbsp; 相对基线 <b>${sign}${it.rel}</b>` +
          padsHtml(it.k9);
      }
      sel.onchange = apply;
      apply();
      return vid;
    };
    const vRef = wire(panes[0], baseIdx, false);
    const vGen = wire(panes[1], ci, true);
    link(vRef, vGen); link(vGen, vRef);
  });
}

function renderChart(){
  const max = Math.max(1, ...ITEMS.map(x => Math.abs(x.rel)));
  const rows = [...ITEMS].sort((a, b) => a.rel - b.rel);
  document.getElementById("chart").innerHTML = rows.map(it => {
    const w = Math.abs(it.rel) / max * 50;               // 半幅百分比
    const neg = it.rel < 0;
    const bar = Math.abs(it.rel) < 0.05 ? "" :
      `<div class="b ${neg ? "n" : "p"}" style="${neg
        ? `right:50%;width:${w}%` : `left:50%;width:${w}%`}"></div>`;
    const sign = it.rel > 0 ? "+" : "";
    return `<div class="crow" data-name="${esc(it.name)}">
      <span class="nm">${esc(it.name)}</span>
      <div class="gut"><div class="axis0"></div>${bar}</div>
      <span class="val">${sign}${it.rel}</span></div>`;
  }).join("");
  document.getElementById("ticks").innerHTML =
    `<span>−${max}</span><span>0</span><span>+${max}</span>`;
  // 点某一行 -> 左边播放位切到它，方便"看数字顺手看画面"
  document.querySelectorAll(".crow").forEach(r => {
    r.onclick = () => {
      const i = ITEMS.findIndex(x => x.name === r.dataset.name);
      const card = document.querySelector(`#cards .card[data-ci="${i}"]`);
      if (card) card.scrollIntoView({behavior: "smooth", block: "center"});
    };
  });
}

function renderTable(){
  const rows = [...ITEMS].sort((a, b) => a.rel - b.rel);
  document.getElementById("tbl").innerHTML =
    `<thead><tr><th>预设</th><th>按键</th><th>动作文本</th>
      <th style="text-align:right">cum_dx</th>
      <th style="text-align:right">相对基线</th></tr></thead><tbody>` +
    rows.map(it => `<tr>
      <td class="nm">${esc(it.name)}</td>
      <td class="nm">${esc(it.keys.join(" ") || "—")}</td>
      <td>${esc(it.text)}</td>
      <td class="num">${it.cum_dx}</td>
      <td class="num">${it.rel > 0 ? "+" : ""}${it.rel}</td></tr>`).join("") + `</tbody>`;
}

// 同卡两边同步：阈值判断防止 seeked 事件互相触发、无限乒乓
function link(a, b){
  a.addEventListener("play", () => {
    if (Math.abs(b.currentTime - a.currentTime) > 0.15) b.currentTime = a.currentTime;
    b.play().catch(() => {});
  });
  a.addEventListener("pause", () => b.pause());
  a.addEventListener("seeked", () => {
    if (Math.abs(b.currentTime - a.currentTime) > 0.15) b.currentTime = a.currentTime;
  });
}

renderStats();
renderCards();
renderChart();
renderTable();
</script>
"""


if __name__ == "__main__":
    main()
