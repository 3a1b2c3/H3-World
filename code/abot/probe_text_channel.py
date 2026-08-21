#!/usr/bin/env python3
"""零训练探针：MiniMax-H3 的文本通道到底控不控画面运动？

这是 docs/action_text_injection_plan.html 第 0 步，整个逐 latent 文本注入方案的红灯/绿灯。

**不加载任何 checkpoint** —— 跑纯基座模型。同一条样本、同一个首帧、同一个 seed，
只改 prompt 尾部一句运动描述，再叠一档 cfg_scale。如果 left / right 两个变体生成出来
几乎一样，说明文本通道本来就不握运动控制权，方案该回头走 PRoPE 那条路；
有明显且方向正确的分叉，才值得往下改 packing 和掩码。

判据不靠肉眼，见 analyze_probe.py：用相位相关估每帧的全局水平位移，
left 与 right 的累计位移应当符号相反。

用法（单卡）:
    python3 code/abot/probe_text_channel.py --variant left --cfg-scale 5.0 \
        --device cuda:0 --vram-limit-gib 30 --allow-busy-gpu
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_abot as IA  # noqa: E402  —— 复用它的 cache 环境、模型路径与 IO 辅助

import argparse  # noqa: E402
import json  # noqa: E402


# 变体表。尾句直接追加在原场景 prompt 之后，其余一切保持不变。
# left / right 是这次探针的主对照：语义完全相反的相机方向。
# none 是基线（原样 prompt，不含任何运动词）；still 是"运动被显式否定"的对照。
VARIANTS = {
    "none":  "",
    "still": " The man stands still and the camera holds steady.",
    "left":  " The man walks forward while the camera pans left.",
    "right": " The man walks forward while the camera pans right.",
}


def load_base_pipeline(device: str, total_gib: float, vram_limit_gib: float | None):
    """加载基座模型：不注册动作层，不挂 LoRA，不读任何 checkpoint。"""
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }
    configs = [
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern=pattern, **vram_config)
        for pattern in (
            "FL2VA/text_encoder/model*.safetensors",
            "FL2VA/transformer/model*.safetensors",
            "FL2VA/video_vae/source/model.safetensors",
            "FL2VA/audio_vae/model.safetensors",
        )
    ]
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=configs,
        processor_config=ModelConfig(
            model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"
        ),
        vram_limit=(vram_limit_gib if vram_limit_gib is not None else max(1.0, total_gib - 2.0)),
    )
    # 基座模型不该有动作层或 LoRA。显式核一遍，避免拿错模型还以为是基座。
    n_lora = sum(len(m.lora_A_weights) for m in pipe.dit.modules()
                 if hasattr(m, "lora_A_weights"))
    if n_lora or getattr(pipe.dit, "num_action_buttons", 0):
        raise RuntimeError(
            f"期望纯基座模型，却发现 lora={n_lora} action_buttons="
            f"{getattr(pipe.dit, 'num_action_buttons', 0)}"
        )
    print(f"基座模型已加载（LoRA 0 对，动作层 0 个）")
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 文本通道零训练探针")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--negative-prompt", default=" ")
    ap.add_argument("--negative-variant", choices=("blank", "still"), default="blank",
                    help="blank=默认空负 prompt（放大的是整段 prompt 的服从度）；"
                         "still=方案设计的「动作零参考」：同一段场景描述 + 静止尾句，"
                         "这样 cfg_scale 放大的恰好是动作那部分的差量")
    ap.add_argument("--sample-id", default=None,
                    help="默认取 selection_seed 选出的第一条，与 step2952/step3000 那批同源")
    ap.add_argument("--metadata", type=Path, default=IA.DEFAULT_METADATA)
    ap.add_argument("--clip-root", type=Path, default=IA.DEFAULT_CLIP_ROOT)
    ap.add_argument("--output-root", type=Path, default=IA.DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--run-name", default="probe_text")
    ap.add_argument("--selection-seed", type=int, default=20260820)
    ap.add_argument("--seed", type=int, default=0, help="生成 seed，所有变体必须一致")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vram-limit-gib", type=float, default=None)
    ap.add_argument("--min-free-gib", type=float, default=110.0)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = IA.read_metadata(args.metadata)
    ids = [args.sample_id] if args.sample_id else []
    row = IA.select_rows(rows, ids, 1, args.selection_seed)[0]
    sample = IA.validate_samples([row], args.clip_root)[0]

    suffix = VARIANTS[args.variant]
    prompt = row["prompt"] + suffix
    negative = (row["prompt"] + VARIANTS["still"] if args.negative_variant == "still"
                else args.negative_prompt)

    out_dir = (IA.ensure_output_root(args.output_root) / IA.safe_run_name(args.run_name)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    neg_tag = "" if args.negative_variant == "blank" else "N"
    tag = f"{args.variant}{neg_tag}_cfg{args.cfg_scale:g}"
    video_path = out_dir / f"{tag}.mp4"
    if video_path.is_file() and not args.overwrite:
        print(f"已存在，跳过: {video_path}")
        return

    print(f"变体      : {args.variant}   cfg_scale={args.cfg_scale:g}   seed={args.seed}")
    print(f"样本      : {row['sample_id']}")
    print(f"尾句      : {suffix.strip() or '（无，基线）'}")
    print(f"负 prompt : {args.negative_variant}  «{negative.strip()[-60:]}»")

    IA.create_cache_dirs()
    torch, total_gib = IA.check_gpu(args.device, args.min_free_gib, args.allow_busy_gpu)
    pipe = load_base_pipeline(args.device, total_gib, args.vram_limit_gib)
    from diffsynth.utils.data.audio_video import write_video_audio

    frames = IA.read_video_frames(sample["video"], IA.NUM_FRAMES)
    gt_path = out_dir / "gt.mp4"
    if not gt_path.is_file():
        IA.write_video_atomic(write_video_audio, video=frames, audio=None, path=gt_path)

    video, audio = pipe(
        prompt=prompt,
        negative_prompt=negative,
        height=IA.HEIGHT,
        width=IA.WIDTH,
        num_frames=IA.NUM_FRAMES,
        num_inference_steps=args.steps,
        seed=args.seed,
        keyframes=[frames[0]],
        keyframe_indices=[0],
        cfg_scale=args.cfg_scale,
    )
    IA.write_video_atomic(write_video_audio, video=video, audio=audio, path=video_path)

    meta_path = out_dir / f"{tag}.json"
    meta_path.write_text(json.dumps({
        "variant": args.variant, "cfg_scale": args.cfg_scale, "seed": args.seed,
        "sample_id": row["sample_id"], "suffix": suffix, "steps": args.steps,
        "negative_prompt": negative, "negative_variant": args.negative_variant, "checkpoint": None,
        "note": "base model, no LoRA, no action conditioning",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成      : {video_path}")


if __name__ == "__main__":
    main()
