#!/usr/bin/env python3
"""Run one action-conditioned generation with a trained checkpoint.

Minimal, single-purpose script: given a checkpoint, a first frame, a scene
description, and a held key combination, it writes one mp4. It does not do
batching, held-out-set sweeps, or report generation -- see README.md for
how to reproduce the paired comparisons shown there.

Usage:
    python3 code/abot/infer.py \\
        --checkpoint output/minimax_h3_abot/7872_directed/step-10000.safetensors \\
        --first-frame path/to/first_frame.png \\
        --scene-prompt "The scene is a cobblestone village street ..." \\
        --action-preset pan-right-fast \\
        --out generated.mp4
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Inference must load the same diffsynth fork that produced the checkpoint:
# the released checkpoint was trained against the directed-mask attention
# semantics in DiffSynth-Studio-h3-v2. Running it against the unmodified
# framework would silently apply a different mask to the same LoRA weights,
# producing a meaningless result.
DIFFSYNTH_ROOT = Path(os.environ.get("ABOT_DIFFSYNTH_ROOT", str(PROJECT_ROOT / "DiffSynth-Studio-h3-v2")))
CACHE_ROOT = PROJECT_ROOT / ".cache"
for _name, _path in {
    "HF_HOME": CACHE_ROOT / "hf",
    "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "TORCH_HOME": CACHE_ROOT / "torch",
    "TORCHINDUCTOR_CACHE_DIR": CACHE_ROOT / "torchinductor",
    "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
    "TMPDIR": CACHE_ROOT / "tmp",
}.items():
    os.environ.setdefault(_name, str(_path))
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(DIFFSYNTH_ROOT / "models")
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(DIFFSYNTH_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# An editable diffsynth install registers a meta_path finder that points
# at the unmodified package regardless of sys.path order. Getting this
# wrong doesn't raise an error -- it silently runs the wrong code and
# produces a plausible-looking but meaningless video -- so it's checked
# explicitly rather than assumed.
if "-v2" in str(DIFFSYNTH_ROOT):
    import inspect

    import diffsynth.models.minimax_h3_dit as _dit
    import diffsynth.pipelines.minimax_h3_audio_video as _pipe

    checks = [
        ("diffsynth resolved from the v2 checkout", "-v2/" in _dit.__file__, _dit.__file__),
        ("directed attention mask present", "leak_out" in inspect.getsource(_dit._build_action_block_masks), ""),
        ("packed-sequence builder is the per-latent version", "action_text_spans_local" in inspect.getsource(_pipe), ""),
    ]
    for name, ok, extra in checks:
        print(f"[preflight] {name}: {'ok' if ok else 'FAILED'} {extra}", flush=True)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("[preflight] the loaded diffsynth is not the expected v2 build; refusing to run")

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import abot_action as A  # noqa: E402
import action_script as S  # noqa: E402

LATENT_T = 37
NUM_FRAMES = 124                # must be 17k+5: 124 (5.2s), 243 (10.1s), 481 (20.0s)
HEIGHT = int(os.environ.get("ABOT_HEIGHT", 480))
WIDTH = int(os.environ.get("ABOT_WIDTH", 832))
if HEIGHT % 32 or WIDTH % 32:
    raise SystemExit(f"height and width must both be multiples of 32, got {WIDTH}x{HEIGHT}")

# The basic, single-combination action vocabulary. Compound and
# out-of-distribution combinations used for the generalization figures in
# the report are not reproduced here; see docs/action_injection_arch.html
# for the full key -> sentence rule table.
ACTION_PRESETS: dict[str, tuple[str, ...]] = {
    "still": (), "forward": ("W",), "back": ("S",),
    "strafe-left": ("A",), "strafe-right": ("D",),
    "tilt-down": ("I",), "tilt-up": ("K",),
    "pan-left": ("J",), "pan-right": ("L",),
    "pan-left-fast": ("J", "F"), "pan-right-fast": ("L", "F"),
}


def load_checkpoint_lora(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    lora = {k: v for k, v in state.items() if ".lora_A." in k or ".lora_B." in k}
    if not lora:
        raise ValueError(f"{path} has no LoRA weights; this script only supports the text-injection checkpoints")
    other = set(state) - set(lora)
    if other:
        raise ValueError(f"{path} has {len(other)} non-LoRA tensors; expected a pure-LoRA checkpoint")
    return lora


def load_pipeline(device: str):
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    dtype = torch.bfloat16
    vram_config = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype,
                       onload_device="cpu", preparing_dtype=dtype, preparing_device=device,
                       computation_dtype=dtype, computation_device=device)
    configs = [
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ]
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=dtype, device=device, model_configs=configs,
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    )
    return pipe


def load_first_frame(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (WIDTH, HEIGHT):
        # Scale to cover the target box on the short side, then center crop --
        # same recipe build_abot_clips.py uses when cutting training clips.
        # A plain resize would distort the image instead.
        scale = max(WIDTH / img.width, HEIGHT / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        left, top = (img.width - WIDTH) // 2, (img.height - HEIGHT) // 2
        img = img.crop((left, top, left + WIDTH, top + HEIGHT))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, help="LoRA checkpoint (.safetensors) from training")
    ap.add_argument("--first-frame", required=True, type=Path, help="image used as the generation's first frame")
    ap.add_argument("--scene-prompt", required=True, help="scene description prefixed to the action clauses")
    ap.add_argument("--action-preset", choices=sorted(ACTION_PRESETS), default="forward",
                    help="held key combination for the whole clip (default: forward)")
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="must be 17k+5")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("generated.mp4"))
    args = ap.parse_args()

    if (args.num_frames - 5) % 17:
        ap.error(f"--num-frames must be 17k+5 (124, 243, 481, ...), got {args.num_frames}")
    latent_t = A.latent_t_for(args.num_frames)

    keys9 = np.zeros((latent_t, len(S.KEYS9)), dtype=np.int64)
    for key in ACTION_PRESETS[args.action_preset]:
        keys9[:, S.KEYS9.index(key)] = 1
    script = S.annotate_from_keys9(keys9)
    print(f"action: {args.action_preset}  ({keys9[0].tolist()})")
    print(f"first latent's sentence: {script[0]}")

    lora_state = load_checkpoint_lora(args.checkpoint)
    lora_pairs = sum(1 for k in lora_state if ".lora_A." in k)
    pipe = load_pipeline(args.device)
    before = sum(len(m.lora_A_weights) for m in pipe.dit.modules() if hasattr(m, "lora_A_weights"))
    pipe.load_lora(pipe.dit, state_dict=lora_state, hotload=True)
    after = sum(len(m.lora_A_weights) for m in pipe.dit.modules() if hasattr(m, "lora_A_weights"))
    if after - before != lora_pairs:
        raise RuntimeError(f"only {after - before}/{lora_pairs} LoRA pairs attached; refusing to run")
    print(f"loaded {lora_pairs} LoRA pairs from {args.checkpoint.name}")

    first_frame = load_first_frame(args.first_frame)
    video, audio = pipe(
        prompt=args.scene_prompt,
        negative_prompt=" ",
        height=HEIGHT, width=WIDTH,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        seed=args.seed,
        keyframes=[first_frame],
        keyframe_indices=[0],
        cfg_scale=args.cfg_scale,
        action_script=script,
        # The "zero action reference" negative prompt: same sentence shape,
        # every clause stationary. This way cfg_scale amplifies the
        # action-induced difference, not generic prompt adherence.
        negative_action_script=S.null_script(latent_t),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    from diffsynth.utils.data.audio_video import write_video_audio
    tmp = args.out.with_name(f"{args.out.stem}.tmp.{os.getpid()}.mp4")
    write_video_audio(video=video, audio=audio, output_path=str(tmp), fps=24, audio_sample_rate=32000)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
