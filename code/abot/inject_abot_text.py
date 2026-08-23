#!/usr/bin/env python3
"""把逐 latent 动作文本条件事后写进 stage 1 的 latent 缓存。

与 inject_abot_action.py 同一套思路（事后注入、原子写、data_id = metadata 行号），
区别是这次改写的是 **prompt_embeds + packed**，而不是加一个 action_cond：

  * 视频/音频 latent 完全不动 —— 换标注方案不需要重跑 stage 1 的 VAE 编码
  * 文本段变成 [原头部（图像 pad + 场景描述） | 37 条标注]，标注按镜像偏移落位
  * `packed` 带出 action_text_rows / action_video_start / action_frame_rows，
    DiT 拿它建硬绑定掩码（标注 k 只与第 k 帧的 video 行互相可见）
  * **删除 action_cond** —— 新方案不再走 FiLM/bias 那条特征空间注入

**头部（首帧视觉 + 场景描述）每次重新编码**，走 `presentation_fl2va(prompt, image_counts)`
—— 与推理侧完全同一次调用，训练/推理由构造保证一致，也不依赖缓存里原有的行
（早期版本沿用缓存行，会因为缓存被改写过而产生两种头部结构）。实测 0.25 s/条。

37 条标注**逐条独立编码**，同一串永远得到同一个嵌入 —— 这既是去重字典的前提，
也是流式推理的硬约束（推理时未来动作未知，没法整段编码）。

幂等：头部每次从源数据重新编码、标注整段重建，与缓存的当前状态无关，
所以重复运行结果稳定，中途失败直接重跑即可。

用法:
    python3 code/abot/inject_abot_text.py --meta data/abot_meta_train_7872.jsonl \\
        --cache output/minimax_h3_abot/7872-cache --dry-run
    python3 code/abot/inject_abot_text.py --meta ... --cache ... --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache"
for _n, _p in {
    "HF_HOME": CACHE_ROOT / "hf", "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "TORCH_HOME": CACHE_ROOT / "torch", "TMPDIR": CACHE_ROOT / "tmp",
}.items():
    os.environ.setdefault(_n, str(_p))
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(PROJECT_ROOT / "DiffSynth-Studio-h3" / "models")
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(PROJECT_ROOT / "DiffSynth-Studio-h3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import abot_action as A  # noqa: E402
import action_script as S  # noqa: E402

LATENT_T = 37
NUM_FRAMES = 124
IMAGE_PAD_ROWS = 392          # 首帧过视觉塔的行数，实测在本数据集上恒定


def collect_scripts(rows, clip_root: Path) -> tuple[list[list[str]], dict[str, None]]:
    """每条样本的 37 句标注，以及去重后的词表。"""
    scripts, vocab = [], {}
    for r in rows:
        pooled = A.bin_to_latent(np.load(clip_root / r["action"])[:NUM_FRAMES], LATENT_T)
        sc = S.annotate_from_keys9(S.keys9(pooled))
        scripts.append(sc)
        for line in sc:
            vocab.setdefault(line, None)
    return scripts, vocab


def scan_pad_used_to(rows, clip_root: Path, tokenizer, align: int = 64) -> tuple[int, dict]:
    """全数据集扫一遍，算出统一的 padding 长度。

    必须统一：flex_attention 是 torch.compile 过的，每换一个序列长度就重编译一次，
    而逐条样本的 text_len 不同，几步就会撞满 Dynamo 的重编译上限。

    这一步很便宜 —— head_len 只用 tokenizer 就能精确算出，不必过视觉塔和编码器：
    presentation_fl2va = 前缀 + 图像 pad(恒定 392) + prompt，三段长度都可直接数。
    """
    from diffsynth.models.minimax_h3_text_encoder import presentation_fl2va

    prefix = len(presentation_fl2va(tokenizer, "", [IMAGE_PAD_ROWS])[0]) - IMAGE_PAD_ROWS
    ann_len: dict[str, int] = {}
    stats = {"used": [], "text_len": []}
    for r in rows:
        pooled = A.bin_to_latent(np.load(clip_root / r["action"])[:NUM_FRAMES], LATENT_T)
        script = S.annotate_from_keys9(S.keys9(pooled))
        for line in script:
            if line not in ann_len:
                ann_len[line] = len(tokenizer(line)["input_ids"])
        head = prefix + IMAGE_PAD_ROWS + len(tokenizer(r["prompt"])["input_ids"])
        text_len = head + sum(ann_len[x] for x in script)
        # cond / audio / video 的行数由 latent 尺寸决定，本数据集恒定
        used = text_len + 390 + 414 + LATENT_T * 390
        stats["used"].append(used)
        stats["text_len"].append(text_len)
    mx = max(stats["used"])
    return ((mx + align - 1) // align) * align, stats


def load_encoder(device: str, dtype=torch.bfloat16):
    """只加载 text_encoder + processor，不碰 DiT / VAE。"""
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    cfg = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype,
               onload_device=device, preparing_dtype=dtype, preparing_device=device,
               computation_dtype=dtype, computation_device=device)
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=[ModelConfig(model_id="MiniMax/MiniMax-H3",
                                   origin_file_pattern="FL2VA/text_encoder/model*.safetensors",
                                   **cfg)],
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3",
                                     origin_file_pattern="FL2VA/processor/"),
    )
    pipe.load_models_to_device(("text_encoder",))
    return pipe


def encode_vocab(pipe, vocab, device: str, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    """标注逐条**独立**编码：同一串永远得到同一个嵌入。"""
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va
    out = {}
    for i, line in enumerate(vocab):
        ids, _tags = presentation_t2va(pipe.tokenizer, line)
        ids = ids.unsqueeze(0).to(device)
        with torch.no_grad():
            h = pipe.text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids))
        out[line] = h.to("cpu", dtype)
        if (i + 1) % 20 == 0 or i + 1 == len(vocab):
            print(f"    编码 {i + 1}/{len(vocab)}", flush=True)
    return out


def encode_head(pipe, row, clip_root: Path, device: str, height=480, width=832):
    """首帧视觉 + 场景描述，走与推理侧同一次 presentation_fl2va 调用。"""
    import imageio.v2 as iio
    from PIL import Image
    from diffsynth.models.minimax_h3_text_encoder import presentation_fl2va
    from diffsynth.pipelines.minimax_h3_audio_video import image_token_counts

    reader = iio.get_reader(str(clip_root / row["video"]), "ffmpeg")
    frame = Image.fromarray(reader.get_data(0)); reader.close()
    frame = frame.convert("RGB").resize((width, height), Image.LANCZOS)
    pv, grid, counts = image_token_counts(pipe.processor, [frame])
    ids, tags = presentation_fl2va(pipe.tokenizer, row["prompt"], counts)
    ids = ids.unsqueeze(0).to(device)
    with torch.no_grad():
        h = pipe.text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                              pixel_values=pv.to(device, pipe.torch_dtype),
                              image_grid_thw=grid.to(device, torch.long))
    return h.to("cpu", pipe.torch_dtype), tags.view(-1).to(torch.long)


def rewrite_one(path: str, script: list[str], head, emb: dict[str, torch.Tensor],
                builder, dry: bool, pad_used_to: int | None = None) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    kwargs, extra = data[0], data[1]
    packed, old_pe = extra["packed"], extra["prompt_embeds"]
    head_rows, head_tags = head
    head_len = int(head_rows.shape[0])
    old_tl = int(packed["text_pos"].numel())

    parts = [head_rows.to(old_pe.dtype)]
    cursor = head_len
    spans = []
    for line in script:
        e = emb[line].to(old_pe.dtype)
        spans.append((cursor, cursor + int(e.shape[0])))
        cursor += int(e.shape[0])
        parts.append(e)
    new_pe = torch.cat(parts, dim=0)
    text_len = int(new_pe.shape[0])

    _, _, lt, lh, lw = kwargs["input_latents"].shape
    audio_t = int(kwargs["audio_input_latents"].shape[-1])
    frame_rows = (lh // 2) * (lw // 2)
    n_key = int(kwargs["keyframe_cond_anchor"].shape[0]) // frame_rows
    keyframe_indices = [0] if n_key == 1 else [0, -1]

    new_packed = builder._build_packed_fl2va(
        text_len, lt, lh, lw, audio_t, keyframe_indices,
        action_text_spans=spans, pad_used_to=pad_used_to)
    tt = new_packed["token_tags"]
    tt[new_packed["text_pos"]] = 1                         # 标注行都是文本
    tt[new_packed["text_pos"][:head_len]] = head_tags      # 头部沿用原标签（图像 pad 走 video 组）

    stat = {"text_len": text_len, "seq_len": int(new_packed["seq_len"]),
            "head_len": head_len, "n_img": int((head_tags == 0).sum()),
            "real_used": int(new_packed["action_real_used"]),
            "old_text_len": old_tl, "old_seq_len": int(packed["seq_len"])}
    if dry:
        return stat

    extra["prompt_embeds"] = new_pe
    extra["packed"] = new_packed
    kwargs.pop("action_cond", None)                        # 新方案不走特征空间注入
    tmp = path + f".tmp.{os.getpid()}"
    torch.save(data, tmp)
    os.replace(tmp, path)
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--clip-root", default=str(PROJECT_ROOT / "data/clips"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0, help="第几个分片（0 起）")
    ap.add_argument("--num-shards", type=int, default=1, help="总分片数；8 卡并行填 8")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条，用于验证")
    ap.add_argument("--pad-used-to", type=int, default=None,
                    help="统一的 padding 长度；不给则自动扫全数据集算出（推荐）")
    ap.add_argument("--dry-run", action="store_true", help="只算不写")
    args = ap.parse_args()

    meta = Path(args.meta)
    if not meta.is_absolute():
        meta = PROJECT_ROOT / meta
    rows = [json.loads(l) for l in open(meta)]
    clip_root = Path(args.clip_root)

    pth = {}
    for root, _, files in os.walk(args.cache):
        for fn in files:
            if fn.endswith(".pth"):
                did = int(fn[:-4])
                if did in pth:
                    sys.exit(f"重复 data_id {did}")
                pth[did] = os.path.join(root, fn)
    ids = sorted(pth)
    if args.limit:
        ids = ids[: args.limit]
    for did in ids:
        if did >= len(rows):
            sys.exit(f"data_id {did} 超出 metadata 行数 {len(rows)}")
    ids = ids[args.shard::args.num_shards]
    tag = f"[{args.shard}/{args.num_shards}] " if args.num_shards > 1 else ""
    print(f"{tag}metadata {len(rows)} 行，缓存 {len(pth)} 条，本分片 {len(ids)} 条")

    scripts, vocab = collect_scripts([rows[i] for i in ids], clip_root)
    print(f"{tag}标注词表 {len(vocab)} 种")

    pipe = load_encoder(args.device)
    pad_used_to = args.pad_used_to
    if pad_used_to is None:
        # 注意：要扫**全部** ids（不是本分片），否则各分片算出的值不一致
        all_ids = sorted(pth)[: args.limit] if args.limit else sorted(pth)
        pad_used_to, sc = scan_pad_used_to([rows[i] for i in all_ids], clip_root, pipe.tokenizer)
        print(f"{tag}统一 padding: used {min(sc['used'])}–{max(sc['used'])} -> {pad_used_to}")
    emb = encode_vocab(pipe, vocab, args.device)
    lens = [int(v.shape[0]) for v in emb.values()]
    print(f"{tag}单条标注 {min(lens)}–{max(lens)} 行（均值 {sum(lens) / len(lens):.1f}）")

    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Unit_PackedSequenceBuilder as U)
    builder = U.__new__(U)

    stats = []
    for n, (did, sc) in enumerate(zip(ids, scripts)):
        head = encode_head(pipe, rows[did], clip_root, args.device)
        stats.append(rewrite_one(pth[did], sc, head, emb, builder, args.dry_run,
                                 pad_used_to=pad_used_to))
        if (n + 1) % 100 == 0 or n + 1 == len(ids):
            print(f"{tag}  {'试算' if args.dry_run else '写入'} {n + 1}/{len(ids)}", flush=True)

    tl = np.array([s["text_len"] for s in stats])
    sl = np.array([s["seq_len"] for s in stats])
    hl = np.array([s["head_len"] for s in stats])
    osl = np.array([s["old_seq_len"] for s in stats])
    print(f"\n{tag}head_len {hl.min()}–{hl.max()}（重新编码，其中图像 pad {stats[0]['n_img']} 行）")
    print(f"{tag}text_len {tl.min()}–{tl.max()}（均值 {tl.mean():.0f}）")
    ru = np.array([s["real_used"] for s in stats])
    print(f"{tag}real_used {ru.min()}–{ru.max()}  ->  统一 padding 到 {pad_used_to}")
    print(f"{tag}seq_len  {osl.min()}–{osl.max()} -> {sl.min()}–{sl.max()}"
          f"（唯一值 {sorted(set(sl.tolist()))}）")
    print(f"{tag}" + ("试算完成，未写盘" if args.dry_run else "写入完成，action_cond 已删除"))


if __name__ == "__main__":
    main()
