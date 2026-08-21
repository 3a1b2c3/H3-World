#!/usr/bin/env python3
"""Run action-conditioned MiniMax-H3 inference on the held-out ABot split.

The script deliberately keeps every writable location below PROJECT_ROOT.  The
default checkpoint search only considers the formal 7872-sample training run;
diagnostic checkpoints must be supplied explicitly with --checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSYNTH_ROOT = PROJECT_ROOT / "DiffSynth-Studio-h3"
CACHE_ROOT = PROJECT_ROOT / ".cache"

# These must be set before importing torch, transformers, modelscope, or diffsynth.
_CACHE_ENV = {
    "HF_HOME": CACHE_ROOT / "hf",
    "HF_HUB_CACHE": CACHE_ROOT / "hf" / "hub",
    "HF_DATASETS_CACHE": CACHE_ROOT / "hf" / "datasets",
    "TRANSFORMERS_CACHE": CACHE_ROOT / "hf" / "transformers",
    "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "TORCH_HOME": CACHE_ROOT / "torch",
    "TORCHINDUCTOR_CACHE_DIR": CACHE_ROOT / "torchinductor",
    "TORCH_EXTENSIONS_DIR": CACHE_ROOT / "torch_extensions",
    "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
    "CUDA_CACHE_PATH": CACHE_ROOT / "cuda",
    "NUMBA_CACHE_DIR": CACHE_ROOT / "numba",
    "CUPY_CACHE_DIR": CACHE_ROOT / "cupy",
    "MODELSCOPE_CACHE": CACHE_ROOT / "modelscope",
    "TMPDIR": CACHE_ROOT / "tmp",
    "PYTHONPYCACHEPREFIX": CACHE_ROOT / "pycache",
}
for _name, _path in _CACHE_ENV.items():
    os.environ[_name] = str(_path)
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(DIFFSYNTH_ROOT / "models")
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.pycache_prefix = str(CACHE_ROOT / "pycache")
sys.path.insert(0, str(DIFFSYNTH_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import html
import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

import abot_action as A
import action_script as S


FORMAL_CHECKPOINT_DIR = PROJECT_ROOT / "output/minimax_h3_abot/7872"
DEFAULT_METADATA = PROJECT_ROOT / "data/abot_meta_test_128.jsonl"
DEFAULT_CLIP_ROOT = PROJECT_ROOT / "data/clips"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output/abot_inference"
NUM_FRAMES = 124
HEIGHT = 480
WIDTH = 832
FPS = 24
LATENT_T = A.latent_t_for(NUM_FRAMES)
EXPECTED_ACTION_DIM = len(A.ACTIVE_KEY_COLS)


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    action_dim: int
    action_mode: str
    action_tensors: int
    lora_tensors: int
    lora_pairs: int
    other_tensors: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ABot 8D first-frame inference with side-by-side GT HTML output."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file. If omitted, use the latest step in output/minimax_h3_abot/7872.",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None, help="Optional output subdirectory name.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Select a specific sample_id; may be repeated. Overrides random selection.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260820)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="文本注入模式下这是动作服从度的放大器：正 prompt 带动作脚本、"
                             "负 prompt 是动作零参考，cfg_scale 放大的恰好是二者之差。"
                             "默认 1.0 等于不开 CFG。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=110.0,
        help="Refuse inference below this free VRAM threshold (default: 110 GiB).",
    )
    parser.add_argument(
        "--vram-limit-gib",
        type=float,
        default=None,
        help="Cap resident model weights (GiB). Default: whole card. Set this when sharing a GPU "
             "with training so DiffSynth streams layers from CPU instead of filling VRAM.",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Bypass the free-VRAM safety check. Use only when GPU sharing is intentional.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate checkpoint, local models, metadata, and [37,8] actions without loading the model.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def latest_formal_checkpoint() -> Path:
    candidates = []
    for path in FORMAL_CHECKPOINT_DIR.glob("step-*.safetensors"):
        match = re.fullmatch(r"step-(\d+)\.safetensors", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"No formal checkpoint in {FORMAL_CHECKPOINT_DIR}. "
            "Training is still in Stage 1, or Stage 2 has not reached its first save step. "
            "For a pipeline-only check, pass --checkpoint "
            "output/minimax_h3_abot/dryrun_firstframe_8d/step-1.safetensors explicitly."
        )
    return max(candidates, key=lambda item: item[0])[1].resolve()


def resolve_checkpoint(explicit: Path | None) -> Path:
    path = latest_formal_checkpoint() if explicit is None else explicit.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def inspect_checkpoint(path: Path) -> CheckpointInfo:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is unavailable in this Python environment. Run with "
            "/opt/dlami/nvme/danze/envs/minimax_h3/bin/python."
        ) from exc

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        # 两种注入模式的权重前缀不同：bias 是 action_embedders.*，
        # film 是 action_scale.* + action_shift.*（见 minimax_h3_dit.enable_action_conditioning）。
        bias_keys = [key for key in keys if key.startswith("action_embedders.")]
        film_keys = [key for key in keys
                     if key.startswith("action_scale.") or key.startswith("action_shift.")]
        if film_keys and bias_keys:
            raise ValueError(f"{path} 同时含 film 和 bias 权重，无法判断模式")
        action_mode = "film" if film_keys else "bias"
        action_keys = film_keys or bias_keys
        lora_keys = [key for key in keys if ".lora_A." in key or ".lora_B." in key]
        other_keys = set(keys) - set(action_keys) - set(lora_keys)
        if not action_keys:
            # 逐 latent 文本注入方案：条件走文本通路，**不建任何动作层**，
            # 所以纯 LoRA 的 checkpoint 是正常形态，不是缺权重。
            if not lora_keys:
                raise ValueError(f"{path} 既无动作层也无 LoRA，不是可用的 checkpoint")
            return CheckpointInfo(str(path), 0, "text", 0, len(lora_keys),
                                  len(lora_keys) // 2, len(other_keys))

        # film 模式下 scale 和 shift 各 50 层，去重后应当仍是 0..49
        layer_ids = sorted({int(key.split(".")[1]) for key in action_keys})
        if layer_ids != list(range(50)):
            raise ValueError(f"Expected action layers 0..49, got {layer_ids[:5]}... ({len(layer_ids)} total)")
        shapes = {tuple(handle.get_slice(key).get_shape()) for key in action_keys}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent action layer shapes: {sorted(shapes)}")
        shape = next(iter(shapes))
        if len(shape) != 2:
            raise ValueError(f"Action embedder must be a matrix, got {shape}")
        action_dim = int(shape[1])

    if action_dim != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"Checkpoint action_dim={action_dim}, but current ABot inference uses "
            f"{EXPECTED_ACTION_DIM}D ({','.join(A.ACTIVE_KEY_COLS)})."
        )
    if not lora_keys:
        raise ValueError("Checkpoint has action weights but no LoRA weights; it is not the current 8D setup.")
    lora_set = set(lora_keys)
    missing_pairs = [
        key for key in lora_keys
        if ".lora_A." in key and key.replace(".lora_A.", ".lora_B.") not in lora_set
    ]
    if missing_pairs or len(lora_keys) % 2:
        raise ValueError(f"Incomplete LoRA A/B pairs in checkpoint: {missing_pairs[:3]}")

    return CheckpointInfo(
        path=str(path),
        action_dim=action_dim,
        action_mode=action_mode,
        action_tensors=len(action_keys),
        lora_tensors=len(lora_keys),
        lora_pairs=len(lora_keys) // 2,
        other_tensors=len(other_keys),
    )


def check_local_models() -> dict[str, int]:
    base = DIFFSYNTH_ROOT / "models/MiniMax/MiniMax-H3/FL2VA"
    required = {
        "text_encoder_shards": list((base / "text_encoder").glob("model-*.safetensors")),
        "transformer_shards": list((base / "transformer").glob("model-*.safetensors")),
        "video_vae": [base / "video_vae/source/model.safetensors"],
        "audio_vae": [base / "audio_vae/model.safetensors"],
        "processor": [base / "processor/tokenizer.json"],
    }
    missing = [str(path) for paths in required.values() for path in paths if not path.is_file()]
    if len(required["text_encoder_shards"]) != 14:
        missing.append(f"expected 14 text-encoder shards, found {len(required['text_encoder_shards'])}")
    if len(required["transformer_shards"]) != 13:
        missing.append(f"expected 13 transformer shards, found {len(required['transformer_shards'])}")
    if missing:
        raise FileNotFoundError("Local MiniMax-H3 model is incomplete:\n  " + "\n  ".join(missing))
    return {name: len(paths) for name, paths in required.items()}


def read_metadata(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("video", "action", "prompt", "sample_id"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number} is missing {key!r}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Metadata is empty: {path}")
    return rows


def select_rows(rows: list[dict], sample_ids: list[str], count: int, seed: int) -> list[dict]:
    if sample_ids:
        by_id = {row["sample_id"]: row for row in rows}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample_id not present in test metadata: {missing}")
        return [by_id[sample_id] for sample_id in sample_ids]
    if count > len(rows):
        raise ValueError(f"Requested {count} samples from a {len(rows)}-row test split")
    return random.Random(seed).sample(rows, count)


def resolve_data_file(clip_root: Path, relative: str) -> Path:
    root = clip_root.expanduser().resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Metadata path escapes clip root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


def load_action(row: dict, clip_root: Path) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    path = resolve_data_file(clip_root, row["action"])
    matrix = np.load(path, allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] < NUM_FRAMES or matrix.shape[1] != A.ACTION_DIM:
        raise ValueError(
            f"{path}: expected at least [{NUM_FRAMES},{A.ACTION_DIM}], got {list(matrix.shape)}"
        )
    pooled = A.bin_to_latent(matrix[:NUM_FRAMES], LATENT_T)
    keys9 = S.keys9(pooled)                      # [latent_t, 9]，第 9 位由 COLMAP 速率导出
    script = S.annotate_from_keys9(keys9)        # 与训练时同一条规则表
    action = pooled[:, A.ACTIVE_KEY_INDICES].astype(np.float32, copy=False)
    if action.shape != (LATENT_T, EXPECTED_ACTION_DIM):
        raise ValueError(f"{path}: binned action shape {action.shape} != {(LATENT_T, EXPECTED_ACTION_DIM)}")
    summary = {
        name: {
            "active_tokens": int(np.count_nonzero(action[:, index] > 0)),
            "total_tokens": LATENT_T,
            "coverage": round(float(np.mean(action[:, index] > 0)), 4),
        }
        for index, name in enumerate(A.ACTIVE_KEY_COLS)
    }
    return action, summary, keys9, script


def validate_samples(rows: list[dict], clip_root: Path) -> list[dict]:
    validated = []
    for row in rows:
        video = resolve_data_file(clip_root, row["video"])
        action, summary, keys9, script = load_action(row, clip_root)
        validated.append({
            "row": row,
            "video": video,
            "action": action,
            "action_summary": summary,
            "keys9": keys9,
            "script": script,
        })
    return validated


def ensure_output_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    allowed = (PROJECT_ROOT / "output").resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Output must stay below {allowed}, got {path}") from exc
    return path


def safe_run_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not value:
        raise ValueError("Run name is empty after sanitization")
    return value


def create_cache_dirs() -> None:
    for path in set(_CACHE_ENV.values()):
        path.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def render_html(run_dir: Path, manifest: dict) -> Path:
    checkpoint = html.escape(Path(manifest["checkpoint"]["path"]).name)
    sections = []
    for index, item in enumerate(manifest["samples"], 1):
        sample_id = html.escape(item["sample_id"])
        prompt = html.escape(item["prompt"])
        action_bits = " ".join(
            f"{name}:{stats['active_tokens']}/{stats['total_tokens']}"
            for name, stats in item["action_summary"].items()
        )
        if item.get("status") == "complete":
            media = f"""
            <div class="videos">
              <figure>
                <figcaption>GT</figcaption>
                <video class="gt" controls loop muted preload="metadata" src="{html.escape(item['gt_video'])}"></video>
              </figure>
              <figure>
                <figcaption>Generated</figcaption>
                <video class="generated" controls loop preload="metadata" src="{html.escape(item['generated_video'])}"></video>
              </figure>
            </div>
            <label class="sync"><input type="checkbox" checked> Sync playback and seeking</label>
            """
        elif item.get("status") == "failed":
            media = f"<p class=\"error\">Inference failed: {html.escape(item.get('error', 'unknown error'))}</p>"
        else:
            media = "<p class=\"pending\">Inference is running...</p>"
        sections.append(f"""
        <section class="sample" data-sample="{sample_id}">
          <div class="sample-head">
            <h2>{index}. {sample_id}</h2>
            <span>seed {item['seed']}</span>
          </div>
          {media}
          <details>
            <summary>Prompt and action coverage</summary>
            <p>{prompt}</p>
            <code>{html.escape(action_bits)}</code>
          </details>
        </section>
        """)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ABot GT / Generated comparison</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #202124; background: #f4f5f6; }}
    header {{ padding: 22px 28px 18px; background: #fff; border-bottom: 1px solid #d9dcdf; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #5b6268; font-size: 14px; }}
    main {{ width: min(1500px, 100%); margin: 0 auto; padding: 20px 24px 48px; }}
    .sample {{ margin: 0 0 18px; padding: 16px; background: #fff; border: 1px solid #d9dcdf; border-radius: 6px; }}
    .sample-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 12px; }}
    h2 {{ margin: 0; font-size: 16px; overflow-wrap: anywhere; letter-spacing: 0; }}
    .sample-head span {{ color: #6b7176; font-size: 13px; white-space: nowrap; }}
    .videos {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    figure {{ min-width: 0; margin: 0; }}
    figcaption {{ margin-bottom: 6px; color: #3f454a; font-size: 13px; font-weight: 650; }}
    video {{ display: block; width: 100%; aspect-ratio: 16 / 9; background: #111; object-fit: contain; }}
    .sync {{ display: inline-flex; align-items: center; gap: 7px; margin-top: 10px; color: #4b5156; font-size: 13px; }}
    details {{ margin-top: 10px; border-top: 1px solid #eceeef; padding-top: 10px; }}
    summary {{ cursor: pointer; color: #3f454a; font-size: 13px; }}
    details p {{ max-width: 1100px; margin: 10px 0; color: #4b5156; font-size: 13px; line-height: 1.55; }}
    code {{ display: block; color: #2f5d41; font-size: 12px; white-space: normal; overflow-wrap: anywhere; }}
    .error {{ margin: 0; padding: 12px; color: #8d2424; background: #fff1f0; border-left: 3px solid #c7443e; }}
    .pending {{ margin: 0; padding: 12px; color: #4b5156; background: #f5f6f7; border-left: 3px solid #858b90; }}
    @media (max-width: 700px) {{ main {{ padding: 14px 10px 36px; }} header {{ padding: 18px 14px; }} .videos {{ gap: 7px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>ABot GT / Generated</h1>
    <p>{len(manifest['samples'])} samples | checkpoint {checkpoint} | 8D W A S D I J K L | first frame only</p>
  </header>
  <main>{''.join(sections)}</main>
  <script>
    document.querySelectorAll('.sample').forEach((section) => {{
      const gt = section.querySelector('video.gt');
      const generated = section.querySelector('video.generated');
      const toggle = section.querySelector('.sync input');
      if (!gt || !generated || !toggle) return;
      let busy = false;
      const syncTime = (from, to) => {{
        if (busy || !toggle.checked || !Number.isFinite(from.currentTime)) return;
        busy = true;
        if (Math.abs(to.currentTime - from.currentTime) > 0.08) to.currentTime = from.currentTime;
        to.playbackRate = from.playbackRate;
        busy = false;
      }};
      [[gt, generated], [generated, gt]].forEach(([from, to]) => {{
        from.addEventListener('play', () => {{ syncTime(from, to); if (toggle.checked) to.play().catch(() => {{}}); }});
        from.addEventListener('pause', () => {{ if (toggle.checked && !busy) to.pause(); }});
        from.addEventListener('seeking', () => syncTime(from, to));
        from.addEventListener('ratechange', () => syncTime(from, to));
        from.addEventListener('timeupdate', () => {{ if (Math.abs(to.currentTime - from.currentTime) > 0.2) syncTime(from, to); }});
      }});
    }});
  </script>
</body>
</html>
"""
    path = run_dir / "index.html"
    tmp = run_dir / "index.html.tmp"
    tmp.write_text(document, encoding="utf-8")
    os.replace(tmp, path)
    return path


def check_gpu(device: str, min_free_gib: float, allow_busy: bool):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(torch.device(device))
    free, total = torch.cuda.mem_get_info(torch.device(device))
    free_gib, total_gib = free / 2**30, total / 2**30
    print(f"GPU {device}: {free_gib:.1f}/{total_gib:.1f} GiB free")
    if not allow_busy and free_gib < min_free_gib:
        raise RuntimeError(
            f"GPU is busy: only {free_gib:.1f} GiB free, below --min-free-gib {min_free_gib:.1f}. "
            "This guard prevents inference from interfering with the current 8-GPU training job."
        )
    return torch, total_gib


def load_pipeline(checkpoint: Path, info: CheckpointInfo, device: str, total_gib: float,
                  vram_limit_gib: float | None = None):
    import torch
    from safetensors.torch import load_file
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
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
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

    state = load_file(str(checkpoint), device="cpu")
    if info.action_mode == "text":
        lora_state = {k: v for k, v in state.items() if ".lora_A." in k or ".lora_B." in k}
        before = sum(len(m.lora_A_weights) for m in pipe.dit.modules() if hasattr(m, "lora_A_weights"))
        pipe.load_lora(pipe.dit, state_dict=lora_state, hotload=True)
        after = sum(len(m.lora_A_weights) for m in pipe.dit.modules() if hasattr(m, "lora_A_weights"))
        if after - before != info.lora_pairs:
            raise RuntimeError(f"只挂上 {after - before}/{info.lora_pairs} 对 LoRA，拒绝推理")
        print(f"文本注入模式：加载 {info.lora_pairs} 对 LoRA，未建动作层")
        return pipe

    def _sub(prefix):
        return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}

    if info.action_mode == "film":
        action_state = {"scale": _sub("action_scale."), "shift": _sub("action_shift.")}
    else:
        action_state = {"bias": _sub("action_embedders.")}
    lora_state = {
        key: value for key, value in state.items()
        if ".lora_A." in key or ".lora_B." in key
    }

    pipe.dit.enable_action_conditioning(info.action_dim, mode=info.action_mode)
    # 这些层是在 VRAM 包装之后建的，权重很小，直接留在计算 GPU 上。
    if info.action_mode == "film":
        pipe.dit.action_scale.load_state_dict(action_state["scale"], strict=True)
        pipe.dit.action_shift.load_state_dict(action_state["shift"], strict=True)
        pipe.dit.action_scale.to(device=pipe.device, dtype=pipe.torch_dtype)
        pipe.dit.action_shift.to(device=pipe.device, dtype=pipe.torch_dtype)
    else:
        pipe.dit.action_embedders.load_state_dict(action_state["bias"], strict=True)
        pipe.dit.action_embedders.to(device=pipe.device, dtype=pipe.torch_dtype)

    before = sum(len(module.lora_A_weights) for module in pipe.dit.modules() if hasattr(module, "lora_A_weights"))
    pipe.load_lora(pipe.dit, state_dict=lora_state, hotload=True)
    after = sum(len(module.lora_A_weights) for module in pipe.dit.modules() if hasattr(module, "lora_A_weights"))
    if after - before != info.lora_pairs:
        raise RuntimeError(
            f"Only {after - before}/{info.lora_pairs} LoRA pairs were attached; refusing unconditioned inference."
        )
    print(f"Loaded 50 action embedders and {info.lora_pairs} LoRA pairs from {checkpoint.name}")
    return pipe


def read_video_frames(path: Path, count: int):
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError(f"{path}: decoded {len(frames)} frames, expected at least {count}")
    return frames


def relative_output_path(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def write_video_atomic(write_video_audio, *, video, audio, path: Path) -> None:
    # 临时名带 pid：多个进程写同一个目标路径时（例如同一条样本的多个变体共用 gt.mp4），
    # 固定的 .tmp.mp4 会让先完成的那个 rename 走文件，后完成的 os.replace 直接 FileNotFound。
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.mp4")
    write_video_audio(
        video=video,
        audio=audio,
        output_path=str(tmp),
        fps=FPS,
        audio_sample_rate=32000,
    )
    os.replace(tmp, path)


def run_inference(
    args: argparse.Namespace,
    checkpoint: Path,
    info: CheckpointInfo,
    samples: list[dict],
) -> Path:
    create_cache_dirs()
    torch, total_gib = check_gpu(args.device, args.min_free_gib, args.allow_busy_gpu)
    pipe = load_pipeline(checkpoint, info, args.device, total_gib, args.vram_limit_gib)
    from diffsynth.utils.data.audio_video import write_video_audio

    output_root = ensure_output_root(args.output_root)
    checkpoint_label = f"{checkpoint.parent.name}_{checkpoint.stem}_sel{args.selection_seed}"
    run_name = safe_run_name(args.run_name or checkpoint_label)
    run_dir = (output_root / run_name).resolve()
    try:
        run_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Run directory escapes output root: {run_dir}") from exc
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "checkpoint": asdict(info),
        "config": {
            "metadata": str(args.metadata.expanduser().resolve()),
            "num_frames": NUM_FRAMES,
            "height": HEIGHT,
            "width": WIDTH,
            "fps": FPS,
            "steps": args.steps,
            "selection_seed": args.selection_seed,
            "generation_seed": args.generation_seed,
            "conditioning": "first_frame+8d_actions",
            "action_columns": A.ACTIVE_KEY_COLS,
        },
        "samples": [],
    }

    for index, sample in enumerate(samples):
        row = sample["row"]
        window = int(row.get("window", 0))
        stem = f"{row['sample_id']}_w{window:03d}"
        sample_dir = run_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        gt_path = sample_dir / "gt.mp4"
        generated_path = sample_dir / "generated.mp4"
        seed = args.generation_seed + index
        item = {
            "sample_id": row["sample_id"],
            "window": window,
            "prompt": row["prompt"],
            "seed": seed,
            "source_video": row["video"],
            "source_action": row["action"],
            "action_summary": sample["action_summary"],
            "gt_video": relative_output_path(gt_path, run_dir),
            "generated_video": relative_output_path(generated_path, run_dir),
            "status": "running",
        }
        manifest["samples"].append(item)
        write_json_atomic(run_dir / "manifest.json", manifest)
        render_html(run_dir, manifest)

        try:
            need_gt = args.overwrite or not gt_path.is_file()
            need_generated = args.overwrite or not generated_path.is_file()
            frames = read_video_frames(sample["video"], NUM_FRAMES) if (need_gt or need_generated) else None
            if need_gt:
                write_video_atomic(
                    write_video_audio,
                    video=frames,
                    audio=None,
                    path=gt_path,
                )
            if need_generated:
                gen_kwargs = dict(
                    prompt=row["prompt"],
                    negative_prompt=args.negative_prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=args.steps,
                    seed=seed,
                    keyframes=[frames[0]],
                    keyframe_indices=[0],
                    cfg_scale=args.cfg_scale,
                )
                action_tensor = None
                if info.action_mode == "text":
                    # 与训练时同一条规则表：9 位按键 -> 标注串。
                    # 负 prompt 用「动作零参考」（全部静止），cfg_scale 放大的
                    # 才是动作那部分的差量，而不是笼统的 prompt 服从度。
                    gen_kwargs["action_script"] = sample["script"]
                    gen_kwargs["negative_action_script"] = S.null_script(LATENT_T)
                else:
                    action_tensor = torch.from_numpy(sample["action"]).to(
                        device=pipe.device, dtype=pipe.torch_dtype
                    )
                    gen_kwargs["action_cond"] = action_tensor
                video, audio = pipe(**gen_kwargs)
                write_video_atomic(
                    write_video_audio,
                    video=video,
                    audio=audio,
                    path=generated_path,
                )
                del action_tensor, video, audio
                torch.cuda.empty_cache()
            item["status"] = "complete"
            print(f"[{index + 1}/{len(samples)}] complete: {stem}")
        except Exception as exc:  # Preserve partial results and continue with independent samples.
            item["status"] = "failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{index + 1}/{len(samples)}] failed: {item['error']}", file=sys.stderr)
        finally:
            torch.cuda.empty_cache()
            write_json_atomic(run_dir / "manifest.json", manifest)
            render_html(run_dir, manifest)

    completed = sum(item["status"] == "complete" for item in manifest["samples"])
    if completed == 0:
        raise RuntimeError(f"All samples failed. See {run_dir / 'manifest.json'}")
    return run_dir / "index.html"


def main() -> None:
    args = parse_args()
    output_root = ensure_output_root(args.output_root)
    if args.run_name is not None:
        safe_run_name(args.run_name)
    checkpoint = resolve_checkpoint(args.checkpoint)
    info = inspect_checkpoint(checkpoint)
    model_files = check_local_models()
    rows = read_metadata(args.metadata)
    selected = select_rows(rows, args.sample_id, args.num_samples, args.selection_seed)
    samples = validate_samples(selected, args.clip_root)

    print(f"Checkpoint : {info.path}")
    print(
        f"Weights    : action={info.action_tensors} ({info.action_dim}D, mode={info.action_mode}), "
        f"LoRA={info.lora_tensors} ({info.lora_pairs} pairs), other={info.other_tensors}"
    )
    print(f"Models     : local and complete ({model_files})")
    print(f"Test split : {args.metadata.expanduser().resolve()} ({len(rows)} rows)")
    print(f"Selected   : {len(samples)} deterministic sample(s), seed={args.selection_seed}")
    for sample in samples:
        active = [
            name for name, stats in sample["action_summary"].items()
            if stats["active_tokens"] > 0
        ]
        print(f"  {sample['row']['sample_id']}  action={sample['action'].shape}  active={','.join(active) or 'none'}")
    print(f"Writable   : {output_root} and {CACHE_ROOT} only")

    if "dryrun" in str(checkpoint).lower():
        print("WARNING: this is a one-step diagnostic checkpoint; it validates the path, not model quality.")
    if args.check_only:
        print("Check-only passed; no model was loaded and no GPU was used.")
        return

    html_path = run_inference(args, checkpoint, info, samples)
    print(f"HTML       : {html_path}")


if __name__ == "__main__":
    main()
