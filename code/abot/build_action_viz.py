#!/usr/bin/env python3
"""把逐 latent 动作标注渲成可视化页：GT 视频 + 37 步时间轴 + 标注，播放时同步高亮。

用途是**肉眼核标注**。规则表的自检只能验词表统计和覆盖度，验不出"这句话跟画面对不对得上"
—— 事实上相机从句漏读平移通道、bin 宽造成的假分档，都是打印真实例子才发现的。

用法:
    python3 code/abot/build_action_viz.py --ids <id> [<id> ...] --out docs/action_prompt_viz.html
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
import abot_action as A
import action_script as S

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data/clips"
META = ROOT / "data/abot_meta_test_128.jsonl"
FFMPEG = (ROOT / "../envs/minimax_h3/lib/python3.10/site-packages/imageio_ffmpeg/"
          "binaries/ffmpeg-linux-x86_64-v7.0.2").resolve()
FPS = 24.156


def camera_category(clause: str) -> str:
    if clause == S.CAMERA_IDLE:
        return "static"
    if clause == S.CAMERA_FOLLOW:
        return "follow"
    if "pans" in clause and "tilts" in clause:
        return "pantilt"
    if "pans" in clause:
        return "pan"
    return "tilt"


def encode_video(src: Path, crf: int = 28) -> str:
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "v.mp4"
        subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)],
                       check=True)
        return "data:video/mp4;base64," + base64.b64encode(dst.read_bytes()).decode()


def build_clip(row: dict) -> dict:
    pooled = A.bin_to_latent(np.load(CLIPS / row["action"])[:124], 37)
    rate = S._camera_rate(pooled)
    k9 = S.keys9(pooled)
    script = S.annotate_from_keys9(k9)

    spans = [e - s for s, e in A.frame_spans(37)]
    starts = np.concatenate([[0], np.cumsum(spans)[:-1]])
    steps = []
    for k in range(37):
        clause = script[k].split(", camera ")[1]
        steps.append({
            "t": round(float(starts[k] / FPS), 3),
            "n": int(spans[k]),
            "k9": [int(x) for x in k9[k]],
            "yaw": round(float(rate[k, 1]), 3),
            "text": script[k],
            "cat": camera_category(clause),
        })
    return {
        "id": row["sample_id"][:8],
        "full": row["sample_id"],
        "video": encode_video(CLIPS / row["video"]),
        "steps": steps,
        "distinct": len(set(script)),
    }


def vocabulary_stats(n_clips: int = 400) -> dict:
    """在训练集上统计词表分布，用于页尾的分类讲解。"""
    from collections import Counter
    rows = [json.loads(l) for l in open(ROOT / "data/abot_meta_train_7872.jsonl")][:n_clips]
    mot, cam, seen = Counter(), Counter(), set()
    total = 0
    for r in rows:
        for line in S.annotate(A.bin_to_latent(np.load(CLIPS / r["action"])[:124], 37)):
            m, c = line[len("the man "):].split(", camera ")
            mot[m] += 1
            cam[c] += 1
            seen.add(line)
            total += 1
    by_cat: dict[str, list] = {}
    for c, n in cam.items():
        by_cat.setdefault(camera_category(c), []).append((c, n / total))
    for v in by_cat.values():
        v.sort(key=lambda x: -x[1])
    return {
        "clips": n_clips, "total": total,
        "motion": [(m, n / total) for m, n in mot.most_common()],
        "camera": {k: v for k, v in sorted(by_cat.items(), key=lambda x: -sum(y[1] for y in x[1]))},
        "n_motion": len(mot), "n_camera": len(cam), "n_vocab": len(seen),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/action_prompt_viz.html")
    args = ap.parse_args()

    rows = {json.loads(l)["sample_id"]: json.loads(l) for l in open(META)}
    clips = []
    for sid in args.ids:
        full = next(k for k in rows if k.startswith(sid))
        print(f"  编码 {full[:8]} …", flush=True)
        clips.append(build_clip(rows[full]))

    print("  统计词表 …", flush=True)
    stats = vocabulary_stats()

    html = TEMPLATE.replace("__CLIPS__", json.dumps(clips, ensure_ascii=False)) \
                   .replace("__STATS__", json.dumps(stats, ensure_ascii=False)) \
                   .replace("__THRESH__", json.dumps({
                       "YAW_MIN": S.YAW_MIN, "YAW_SHARP": S.YAW_SHARP,
                       "PITCH_MIN": S.PITCH_MIN, "TRANS_MIN": S.TRANS_MIN,
                       "ANCHOR": S.SUBJECT_ANCHOR}, ensure_ascii=False))
    args.out.write_text(html, encoding="utf-8")
    print(f"页面: {args.out}  ({args.out.stat().st_size / 2**20:.1f} MB)")


TEMPLATE = r"""<title>动作标注核对台</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
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
h3{font-size:16px;font-weight:600;color:var(--ink);margin:26px 0 6px}
p{max-width:70ch}
code{font-family:var(--mono);font-size:.87em;background:var(--sunken);border:1px solid var(--line);
     border-radius:3px;padding:1px 5px;color:var(--ink)}
.sub{color:var(--muted);max-width:72ch;margin:0 0 18px}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;
       background:var(--line);border:1px solid var(--line);border-radius:7px;overflow:hidden;margin:0 0 10px}
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

.stage{max-width:780px;margin:0 auto;padding:20px 18px 4px}
video{width:100%;display:block;border-radius:7px;background:#000}
.strip{display:flex;gap:1px;height:22px;margin-top:10px;border-radius:4px;overflow:hidden;
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
.row .k{font-family:var(--mono);font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
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
td.num{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums;text-align:right;
       color:var(--ink);white-space:nowrap}
.bar{display:inline-block;height:8px;border-radius:2px;background:var(--accent);opacity:.75;
     vertical-align:middle;margin-left:8px}

pre{background:var(--sunken);border:1px solid var(--line);border-radius:7px;padding:14px 16px;
    overflow-x:auto;font-family:var(--mono);font-size:12.5px;line-height:1.6;color:var(--ink);margin:14px 0}
pre .c{color:var(--muted)}
.note{border-left:3px solid var(--accent);background:var(--surface);padding:14px 18px;
      margin:18px 0;border-radius:0 7px 7px 0}
.note p{margin:0;font-size:14.5px}
.note p+p{margin-top:8px}
.note b{color:var(--ink)}
footer{margin-top:52px;padding-top:16px;border-top:1px solid var(--line);
       font-family:var(--mono);font-size:11.5px;color:var(--muted)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">MiniMax-H3 · ABot 动作条件微调</p>
  <h1>动作标注核对台</h1>
  <p class="lede">每个 latent 一条结构化文本条件，是 <b>9 位按键的纯函数</b> ——
  原有 8 键加一位 <code>F</code>(fast)，训练与推理因此走完全同一条路径。
  这页拿来<b>肉眼核对标注对不对得上画面</b>：播放视频，当前 latent 的按键位与标注会同步高亮。
  规则表的自检验得了词表统计，验不了这个。</p>
  <div class="stats" id="stats"></div>
  <div class="legend" id="legend"></div>
</header>

<div id="clips"></div>

<h2>9 位按键怎么映射到标注</h2>
<p class="sub">模板固定，只换两个槽：<code>the man &lt;怎么动&gt;, camera &lt;相机视角怎么动&gt;</code>。
两个槽都是这 9 位的<b>确定性函数</b> —— 同样的按键永远得到同样的串。
这既是逐条独立编码 + 去重字典的前提，也让训练与推理不会出现表示上的失配。</p>

<div class="scroll"><table>
  <thead><tr><th>槽</th><th>输入位</th><th>规则</th></tr></thead>
  <tbody>
    <tr><td class="nm" rowspan="2">槽一<br>怎么动</td><td class="nm">W S A D</td>
        <td>按 W/S/A/D 固定次序拼接：<code>walks forward</code> / <code>walks backward</code> /
            <code>strafes left</code> / <code>strafes right</code>，用 <code>and</code> 连接</td></tr>
    <tr><td class="nm">—</td><td>全部为 0 → <code>stands still</code>。
        <b>对轴净化</b>：W∧S 相消、A∧D 相消（<code>bin_to_latent</code> 对按键取 amax，
        4 帧窗口里先按 W 后按 S 会让两位同时为 1）</td></tr>
    <tr><td class="nm" rowspan="4">槽二<br>相机</td><td class="nm">J / L</td>
        <td><code>pans left</code> / <code>pans right</code>，J∧L 相消</td></tr>
    <tr><td class="nm">F</td>
        <td>摇镜的速度档：F=1 → <code>sharply</code>，F=0 → <code>slowly</code></td></tr>
    <tr><td class="nm">I / K</td>
        <td><code>tilts down</code> / <code>tilts up</code>，I∧K 相消；可与摇镜叠加</td></tr>
    <tr><td class="nm">无相机键</td>
        <td>有移动键 → <code>follows him</code>；无移动键 → <code>holds steady</code></td></tr>
  </tbody>
</table></div>

<h3>第 9 位 F 是从哪来的，为什么非要它不可</h3>
<p>原始录像只有 8 个键。<b>方向能从 IJKL 读出来，速度读不出来</b> —— 600 条 clip 的实测：</p>
<div class="scroll"><table>
  <thead><tr><th>问题</th><th>实测</th><th>结论</th></tr></thead>
  <tbody>
    <tr><td>按 J 的步里 slowly / sharply 各占多少</td><td class="num">0.66 / 0.34</td>
        <td>接近抛硬币，键里没有这个信息</td></tr>
    <tr><td>速度会不会随按住时长变化</td>
        <td class="num">第0步 0.122 → 第1–5步 0.203 → 第6步+ 0.176</td>
        <td>第 0 步偏低只是分箱边界效应（键在 bin 中途按下），之后就平台了</td></tr>
    <tr><td>速度是不是片段级常量</td><td class="num">ICC ≈ 0.475</td>
        <td>一半方差在片段之间、一半在片段之内，整段档位一致的只占 38.9%</td></tr>
  </tbody>
</table></div>
<p>所以 F 由 <b>COLMAP 实测摇镜速率</b>合成：按了 J 或 L 且逐帧 <code>|d_yaw| ≥ 0.225</code> 时置 1。
推理时这一位由用户直接按。众数查表预测器的准确率因此从 <b>0.700 → 0.871</b>。</p>

<div class="note">
<p><b>为什么不干脆丢掉速度档、只留 4 个相机键。</b>那样查表准确率还能再高一点（0.890），
但相机从句会退化成 IJKL 的确定性函数 —— COLMAP 那 6 个通道就白测了，
方案又回到"把 one-hot 换成英文，信息量一个比特没多"。多这一位，把测出来的速度保住了。</p>
<p><b>被舍弃的一类。</b>原来还有 <code>drifts right/forward/…</code>（角色没动而相机自己在平移，
占 1.7%）。它按定义无法从按键推出，推理时也给不出来，所以并进 <code>holds steady</code>。
这是训练与推理之间一个已知且不可消除的小失配。</p>
</div>

<h3>槽一 · 移动从句的实际分布 <span style="color:var(--muted);font-weight:400" id="nmot"></span></h3>
<div class="scroll"><table id="tmot"></table></div>

<h3>槽二 · 相机从句的实际分布 <span style="color:var(--muted);font-weight:400" id="ncam"></span></h3>
<div class="scroll"><table id="tcam"></table></div>

<footer>
minimax_finetune · 规则表 code/abot/action_script.py · 方案 docs/action_text_injection_plan.html
</footer>
</div>

<script>
const CLIPS = __CLIPS__, STATS = __STATS__, THRESH = __THRESH__;
const CATNAME = {pan:"摇镜", pantilt:"摇镜+俯仰", tilt:"俯仰",
                 follow:"跟随", static:"静止"};
const CATORDER = ["pan","pantilt","tilt","follow","static"];
const KEYS9 = ["W","A","S","D","I","J","K","L","F"];
const cv = c => `var(--c-${c})`;
const esc = s => s.replace(/[&<>]/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[m]));

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
  [9, "位按键"], [37, "latent / 片段"],
  [STATS.n_motion + "×" + STATS.n_camera, "移动 × 相机从句"],
  [STATS.n_vocab, "种标注（去重）"],
].map(([b,s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");

document.getElementById("legend").innerHTML = CATORDER
  .map(c => `<span><i style="background:${cv(c)}"></i>${CATNAME[c]}</span>`).join("");

document.getElementById("nmot").textContent = `· ${STATS.n_motion} 种`;
document.getElementById("ncam").textContent = `· ${STATS.n_camera} 种，六类`;

document.getElementById("clips").innerHTML = CLIPS.map((c, ci) => {
  const total = c.steps.reduce((a, s) => a + s.n, 0);
  const strip = c.steps.map((s, k) =>
    `<div data-c="${ci}" data-k="${k}" title="步 ${k} · ${esc(s.text)}"
          style="flex:${s.n};background:${cv(s.cat)}"></div>`).join("");
  const rows = c.steps.map((s, k) => {
    const [mot, cam] = s.text.slice("the man ".length).split(", camera ");
    const bits = padsHtml(s.k9, true);
    return `<div class="row" data-c="${ci}" data-k="${k}">
      <span class="k">${k} · ${s.t.toFixed(2)}s</span>
      ${bits}
      <span class="num">${s.yaw >= 0 ? "+" : ""}${s.yaw.toFixed(3)}</span>
      <span class="txt"><i class="dot" style="background:${cv(s.cat)}"></i>the man <b>${esc(mot)}</b>, camera <b>${esc(cam)}</b></span>
    </div>`;
  }).join("");
  return `<div class="card">
    <div class="card-head">
      <span class="idx">${String(ci + 1).padStart(2, "0")}</span>
      <span class="sid">${c.id}</span>
      <span class="chip">${c.distinct} 种 / 37 步</span>
      <span class="spacer"></span>
      <span class="chip">124 帧 · ${(total / 24.156).toFixed(2)}s</span>
    </div>
    <div class="stage">
      <video src="${c.video}" controls loop muted playsinline data-c="${ci}"></video>
      <div class="strip" data-c="${ci}">${strip}</div>
      <div class="axis"><span>0.00s</span><span>latent 0 → 36</span><span>${(total / 24.156).toFixed(2)}s</span></div>
      <div class="now" id="now-${ci}"></div>
    </div>
    <div class="rows" data-c="${ci}">${rows}</div>
  </div>`;
}).join("");

// 播放时同步高亮当前 latent
function highlight(ci, k) {
  const st = CLIPS[ci].steps[k];
  const [mot, cam] = st.text.slice("the man ".length).split(", camera ");
  const now = document.getElementById(`now-${ci}`);
  if (now) now.innerHTML =
    `${padsHtml(st.k9)}
     <span class="txt"><i class="dot" style="background:${cv(st.cat)}"></i>the man <b>${esc(mot)}</b>, camera <b>${esc(cam)}</b></span>
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
document.querySelectorAll("video").forEach(v => {
  const ci = +v.dataset.c;
  v.addEventListener("timeupdate", () => highlight(ci, stepAt(ci, v.currentTime)));
});
// 点时间轴或某一行都能跳到那一步
document.addEventListener("click", e => {
  const el = e.target.closest("[data-k]");
  if (!el) return;
  const ci = +el.dataset.c, k = +el.dataset.k;
  const v = document.querySelector(`video[data-c="${ci}"]`);
  if (v) { v.currentTime = CLIPS[ci].steps[k].t; highlight(ci, k); }
});

// 页尾表格
const pct = x => (x * 100).toFixed(1) + "%";
const barw = x => Math.max(2, x * 260).toFixed(0);
document.getElementById("tmot").innerHTML =
  `<thead><tr><th>移动从句</th><th style="text-align:right">占比</th><th></th></tr></thead><tbody>` +
  STATS.motion.map(([m, f]) => `<tr><td class="nm">the man ${esc(m)}</td>
    <td class="num">${pct(f)}</td>
    <td><span class="bar" style="width:${barw(f)}px"></span></td></tr>`).join("") + `</tbody>`;

document.getElementById("tcam").innerHTML =
  `<thead><tr><th>类别</th><th>相机从句</th><th style="text-align:right">占比</th><th></th></tr></thead><tbody>` +
  CATORDER.filter(c => STATS.camera[c]).map(c => {
    const list = STATS.camera[c], sum = list.reduce((a, x) => a + x[1], 0);
    return list.map(([txt, f], i) => `<tr>
      ${i === 0 ? `<td rowspan="${list.length}" class="nm"><i class="dot" style="background:${cv(c)}"></i>${CATNAME[c]}<br>
        <span style="color:var(--muted);font-size:11.5px">${pct(sum)} · ${list.length} 种</span></td>` : ""}
      <td class="nm" style="color:var(--body)">camera ${esc(txt)}</td>
      <td class="num">${pct(f)}</td>
      <td><span class="bar" style="width:${barw(f)}px;background:${cv(c)}"></span></td></tr>`).join("");
  }).join("") + `</tbody>`;

const _unused_thr =
  `<thead><tr><th>常量</th><th style="text-align:right">值</th><th>含义</th></tr></thead><tbody>` +
  [["YAW_MIN", THRESH.YAW_MIN, "逐帧 |d_yaw| 低于此值不写摇镜"],
   ["YAW_SHARP", THRESH.YAW_SHARP, "高于此值写 sharply，之间写 slowly"],
   ["PITCH_MIN", THRESH.PITCH_MIN, "逐帧 |d_pitch| 高于此值才写 tilts"],
   ["TRANS_MIN", THRESH.TRANS_MIN, "逐帧平移 |max| 高于此值算相机在动"]]
  .map(([k, v, d]) => `<tr><td class="nm">${k}</td><td class="num">${v}</td><td>${d}</td></tr>`).join("") +
  `<tr><td class="nm">主体锚点</td><td class="num">—</td><td>全局一条：<code>${esc(THRESH.ANCHOR)}</code></td></tr>` +
  `</tbody>`;
</script>
"""

if __name__ == "__main__":
    main()
