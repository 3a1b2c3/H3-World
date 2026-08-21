#!/usr/bin/env python3
"""ABot-World-Explorer-500h 原始数据 -> MiniMax-H3 训练用切片 + metadata。

一条 episode 是 60s / 1800 帧 / 1920x1080 / 30fps / 111 MB，而 H3 一次只吃
124 帧 @24fps @832x480。所以必须先切片，顺带把三件事一次做完：

  1. **30fps -> 24fps**：每 5 个源帧丢第 5 个（精确 24fps，ffmpeg select 可精确表达）。
     切出来的片子直接就是 24fps，于是 stage 1 的 `LoadVideo(fix_frame_rate=True,
     frame_rate=24)` 读回来是**恒等映射** —— 实测 0 重复帧、不丢尾帧。
     对比 V-Rising（16fps 源上采样到 24fps）：35/107 = 32.7% 是复制帧、尾部丢 0.56s。
  2. **1920x1080 -> 832x480**：scale 到短边 480 再中心裁。切完就是训练分辨率，
     stage 1 里的 ImageCropAndResize 退化成恒等，不再二次重采样。
  3. **动作对齐**：视频帧和动作用**同一个显式源帧号列表**取，按构造逐帧对齐，
     不依赖对 H3 内部 round() 映射的推断。这直接关掉了世界模型文档 §7.0d
     那条「时序对齐仍未验」的风险 —— 这里它是构造性成立的，且有 --verify 复核。

行序纪律（姊妹文档 §7.4b 的教训）：所有 tier 都是同一个全局顺序的**前缀**，
所以 smoke(64) 的缓存可以直接被 pilot(2000) 复用，pilot 的可以被 main(8000) 复用。
扩量时只能往后追加，绝不能重排前缀。

用法:
    python3 build_abot_clips.py --plan                      # 只打印计划，不动盘
    python3 build_abot_clips.py --num-clips 64   --workers 16
    python3 build_abot_clips.py --num-clips 2000 --workers 24
    python3 build_abot_clips.py --verify 8                  # 抽查对齐
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abot_action as A

SRC_ROOT = os.environ.get("ABOT_SRC_ROOT", "/nfs/yixinyang/code/LongLive/data/ABot-World-Explorer-500h")  # 本机未挂载，重建切片时用 ABOT_SRC_ROOT 指到 HF 下载目录
OUT_ROOT = os.environ.get("ABOT_OUT_ROOT", "/opt/dlami/nvme/danze/minimax_finetune/data")
CLIP_DIR = f"{OUT_ROOT}/clips"

NUM_FRAMES = 124          # 17*7+5，H3 原生值（所有官方 example 都是 124）
PAD = 6                   # 多写几帧，保证 floor(duration*24) 绝不会掉到 124 以下
N_OUT = NUM_FRAMES + PAD  # 每个切片实际写 130 帧
HEIGHT, WIDTH = 480, 832
SEED = 20260817


def resolve_ffmpeg() -> str:
    configured = os.environ.get("ABOT_FFMPEG")
    if configured:
        return configured
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("ffmpeg not found; set ABOT_FFMPEG to its executable path") from exc


FFMPEG = resolve_ffmpeg()


# --------------------------------------------------------------------------- #
def episode_order():
    """全局固定的 episode 顺序。只依赖 sample_id 与 SEED，任何机器上可复现。"""
    ids = []
    for prefix in sorted(os.listdir(f"{SRC_ROOT}/data")):
        for sid in sorted(os.listdir(f"{SRC_ROOT}/data/{prefix}")):
            ids.append(sid)
    random.Random(SEED).shuffle(ids)
    return ids


def clip_plan(n_clips: int, ids):
    """(sample_id, window_index) 的全局顺序：先把每条 episode 的第 0 窗排完，
    再排第 1 窗。这样「加窗口」和「加 episode」都是纯追加，前缀永远保序。"""
    plan, w = [], 0
    while len(plan) < n_clips:
        for sid in ids:
            plan.append((sid, w))
            if len(plan) >= n_clips:
                break
        w += 1
    return plan


def window_start(sid: str, w: int, total_frames: int, span: int) -> int:
    """第 w 个窗口的源帧起点。

    把 episode 切成 n_slots 个互不重叠的槽位，用 sample_id 播种打乱槽位顺序，
    第 w 个窗口取打乱后的第 w 个槽。两个作用：
      * 同一条 episode 的多个窗口天然不重叠（否则相邻窗口近乎重复样本）；
      * **窗口 0 均匀散布在整条 60s 上**，而不是所有 episode 都从开头切
        —— 开头往往是镜头刚起步、动作稀疏的一段，全取开头会让动作分布偏斜。
    """
    usable = total_frames - span
    if usable < 0:
        raise ValueError(f"episode 只有 {total_frames} 帧，放不下 {span} 帧的窗口")
    n_slots = max(1, total_frames // span)
    rng = random.Random(f"{sid}:{SEED}")
    slots = list(range(n_slots))
    rng.shuffle(slots)
    slot = slots[w % n_slots]
    jitter = random.Random(f"{sid}:{w}:jit").randrange(0, max(1, span // 2))
    return min(usable, slot * span + jitter)


# --------------------------------------------------------------------------- #
MIN_PROMPT_WORDS = 30


def pick_prompt(cap: dict) -> str:
    """选 prompt。默认 `scene_static`（窗口无关，见文档 §3.5），退化时回退 `narrative`。

    源 caption 里确实存在脏数据：8000 条里有 1 条的 `scene_static` 是字面量 `'!!!'`。
    这种条目不值得为它破坏行序去删（删一行 = 后面全部错位 = 缓存全废），
    但也不该拿 `'!!!'` 去训 —— 回退到 `narrative` 是无代价的。
    `narrative` 描述整条 60s 的运动、与窗口不符，所以只在退化时用。
    """
    s = (cap.get("scene_static") or "").strip()
    if len(s.split()) >= MIN_PROMPT_WORDS:
        return s
    return (cap.get("narrative") or "").strip() or s


def build_one(task):
    sid, w, crf = task
    prefix = sid[:2]
    ep_dir = f"{SRC_ROOT}/data/{prefix}/{sid}"
    out_dir = f"{CLIP_DIR}/{prefix}"
    stem = f"{sid}_w{w:03d}"
    mp4 = f"{out_dir}/{stem}.mp4"
    npy = f"{out_dir}/{stem}.npy"
    try:
        ep = A.read_episode(f"{ep_dir}/annotations.tar")
        if ep["control_scheme"] != "WASD_QE_locomotion_IJKL_rotation":
            return dict(sid=sid, w=w, ok=False, err=f"control_scheme={ep['control_scheme']}")
        span = A.window_span(N_OUT)
        s = window_start(sid, w, ep["total_frames"], span)

        os.makedirs(out_dir, exist_ok=True)
        if not (os.path.exists(mp4) and os.path.exists(npy)):
            scale = A.episode_translation_scale(ep)
            act = A.window_action_matrix(ep, s, N_OUT, scale)
            np.save(npy + ".tmp.npy", act)
            os.replace(npy + ".tmp.npy", npy)

            vf = (f"select='between(n\\,{s}\\,{s + span - 1})"
                  f"*not(eq(mod(n-{s}\\,5)\\,4))'"
                  f",setpts=N/24/TB,scale=-2:{HEIGHT},crop={WIDTH}:{HEIGHT}")
            tmp = mp4 + ".tmp.mp4"
            subprocess.run(
                [FFMPEG, "-y", "-v", "error", "-i", f"{ep_dir}/video.mp4",
                 "-vf", vf, "-r", "24", "-frames:v", str(N_OUT),
                 "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
                 "-pix_fmt", "yuvj420p", "-an", tmp],
                check=True, capture_output=True)
            os.replace(tmp, mp4)          # 同目录 rename 原子，文件在即完整

        cap = ep["caption"]
        return dict(sid=sid, w=w, ok=True, src_start=s,
                    video=f"{prefix}/{stem}.mp4", action=f"{prefix}/{stem}.npy",
                    prompt=pick_prompt(cap),
                    narrative=cap.get("narrative", "").strip(),
                    perspective=cap.get("perspective"))
    except subprocess.CalledProcessError as e:
        return dict(sid=sid, w=w, ok=False, err=f"ffmpeg: {e.stderr.decode()[:200]}")
    except Exception as e:                                    # noqa: BLE001
        return dict(sid=sid, w=w, ok=False, err=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
def verify(n: int, shifts=(-2, -1, 0, 1, 2)):
    """抽查切片与源视频的逐帧对齐。

    判据是**位移扫描的 argmin**，不是绝对像素差。理由：我这里的参考帧用
    PIL bilinear 复现 ffmpeg 的 bicubic 缩放，本身就有 5~6/255 的地板噪声，
    而慢镜头下相邻帧差可能也只有这个量级 —— 拿「对齐 vs 错一帧」的绝对值
    比大小会得出没有分辨力的结论。但那个地板噪声对所有 shift 是**共同的**，
    所以 argmin 落在 0 仍然是干净的判据。
    """
    import imageio.v2 as imageio
    import PIL.Image
    sys.path.insert(0, os.environ.get("DIFFSYNTH_ROOT", "/opt/dlami/nvme/danze/minimax_finetune/DiffSynth-Studio-h3"))
    from diffsynth.core.data.operators import LoadVideo

    rows = [json.loads(l) for l in open(f"{OUT_ROOT}/abot_manifest.jsonl")]
    picks = random.Random(0).sample(rows, min(n, len(rows)))
    offs = A.window_offsets(N_OUT)
    probes = [10, 40, 80, 118]
    lv = LoadVideo(num_frames=NUM_FRAMES, time_division_factor=17,
                   time_division_remainder=5, frame_rate=24, fix_frame_rate=True)
    print(f"{'sample':14s} {'nfrm':>5s} {'dup':>4s}  " +
          "  ".join(f"shift{k:+d}" for k in shifts) + "   argmin  verdict")
    n_pass = 0
    for r in picks:
        sid, s = r["sample_id"], r["src_start"]
        frames = [np.asarray(f).astype(np.float32) for f in lv(f"{CLIP_DIR}/{r['video']}")]
        dup = sum(1 for a, b in zip(frames, frames[1:]) if np.array_equal(a, b))

        want = {s + offs[j + k] for j in probes for k in shifts}
        got, hi = {}, max(want)
        rd = imageio.get_reader(f"{SRC_ROOT}/data/{sid[:2]}/{sid}/video.mp4")
        for i, fr in enumerate(rd):
            if i in want:
                im = PIL.Image.fromarray(fr).resize((854, WIDTH and 480), PIL.Image.BILINEAR)
                got[i] = np.asarray(im)[:, 11:11 + WIDTH].astype(np.float32)
            if i >= hi:
                break
        rd.close()

        scores = [float(np.mean([np.abs(got[s + offs[j + k]] - frames[j]).mean()
                                 for j in probes])) for k in shifts]
        amin = shifts[int(np.argmin(scores))]
        ok = len(frames) == NUM_FRAMES and dup == 0 and amin == 0
        n_pass += ok
        print(f"{sid[:12]:14s} {len(frames):5d} {dup:4d}  " +
              "  ".join(f"{v:6.2f}" for v in scores) +
              f"   {amin:+d}     {'PASS' if ok else 'FAIL'}")
    print(f"\n{n_pass}/{len(picks)} 通过（124 帧 / 0 重复帧 / 位移扫描 argmin=0）")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--tiers", default="64,2000,8000",
                    help="额外导出的前缀 tier，逗号分隔")
    ap.add_argument("--plan", action="store_true", help="只打印计划")
    ap.add_argument("--verify", type=int, default=0, help="抽查 N 条对齐后退出")
    args = ap.parse_args()

    if args.verify:
        verify(args.verify)
        return

    span = A.window_span(N_OUT)
    ids = episode_order()
    plan = clip_plan(args.num_clips, ids)
    print(f"源: {SRC_ROOT}  ({len(ids)} episodes)")
    print(f"出: {CLIP_DIR}")
    print(f"切片: {N_OUT} 帧写盘 / 训练用前 {NUM_FRAMES} 帧 @24fps "
          f"({NUM_FRAMES / 24:.3f}s), {WIDTH}x{HEIGHT}, 每片吃 {span} 源帧")
    print(f"latent_t = {A.latent_t_for(NUM_FRAMES)}   action_dim = {A.ACTION_DIM}")
    print(f"计划 {len(plan)} 片，最大窗口序号 {max(w for _, w in plan)}，workers={args.workers}")
    if args.plan:
        for sid, w in plan[:5]:
            print(f"   {sid} w{w:03d}")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    t0 = time.time()
    results = {}
    tasks = [(sid, w, args.crf) for sid, w in plan]
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(build_one, t): (t[0], t[1]) for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results[(r["sid"], r["w"])] = r
            if i % 50 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  {i}/{len(tasks)}  {el:.0f}s  {el / i:.2f}s/片  "
                      f"ETA {(len(tasks) - i) * el / i / 60:.1f}min", flush=True)

    ok = [results[k] for k in plan if results[k]["ok"]]
    bad = [results[k] for k in plan if not results[k]["ok"]]
    man = f"{OUT_ROOT}/abot_manifest.jsonl"
    with open(man, "w") as f:
        for r in ok:
            f.write(json.dumps({
                "video": r["video"], "input_audio": r["video"], "prompt": r["prompt"],
                "action": r["action"], "sample_id": r["sid"], "window": r["w"],
                "src_start": r["src_start"], "perspective": r["perspective"],
                "narrative": r["narrative"],
            }, ensure_ascii=False) + "\n")
    print(f"\n成功 {len(ok)}  失败 {len(bad)}  ->  {man}")
    for r in bad[:10]:
        print(f"   FAIL {r['sid']} w{r['w']}: {r['err']}")

    lines = open(man).read().splitlines()
    for tier in [int(x) for x in args.tiers.split(",") if x]:
        if tier > len(lines):
            print(f"   tier {tier} 超过成功数 {len(lines)}，跳过")
            continue
        p = f"{OUT_ROOT}/abot_meta_{tier}.jsonl"
        with open(p, "w") as f:
            f.write("\n".join(lines[:tier]) + "\n")
        print(f"   tier {tier:6d} -> {p}")
    print("\n⚠️  所有 tier 都是 manifest 的前缀，缓存可跨 tier 复用（姊妹文档 §7.4b）。"
          "\n   扩量只能追加，不要重排前缀，否则 {data_id}.pth 全部错位。")


if __name__ == "__main__":
    main()
