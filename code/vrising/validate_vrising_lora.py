"""V-Rising LoRA 验证：同 seed / 同首尾帧，对比 base vs LoRA。

只跑通训练脚本（exit 0）不能说明模型学到了东西，所以这里做 A/B：
同一条 clip 的首尾帧 + 同一个 prompt + 同一个 seed，先用原模型生成一遍，
再挂上 LoRA 生成一遍。两段视频有可见差异才说明 LoRA 真的起作用了。

用法:
    CUDA_VISIBLE_DEVICES=1 python3 /nfs/danze/validate_vrising_lora.py
    CUDA_VISIBLE_DEVICES=1 python3 /nfs/danze/validate_vrising_lora.py --steps 20   # 快速看
"""
import argparse, json, os, sys, time

sys.path.insert(0, "/nfs/danze/repo/DiffSynth-Studio-new")
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.utils.data import VideoData

DATA_BASE = "/nfs/danze/data/v_rising/vrising_train_bundle/data/transform_classified"
HEIGHT, WIDTH, NUM_FRAMES = 480, 832, 107   # 必须 17n+5，训练时也是 107

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--index", type=int, default=0, help="用子集里第几条")
# 子集 / checkpoint / 输出目录可配：Stage A 已经从 smoke 推进到 form_wolf，
# 每个 subset 的 LoRA 存在自己的 output_path 下，写死路径就只能验 smoke 那一轮。
p.add_argument("--subset", default="smoke", help="用哪个 h3_meta_<subset>.jsonl 取首尾帧和 prompt")
p.add_argument("--lora", default=None, help="LoRA checkpoint 路径，默认取 smoke 的 epoch-4")
p.add_argument("--outdir", default=None, help="输出目录，默认 /nfs/danze/eval/vrising_<subset>")
args = p.parse_args()

META = f"{DATA_BASE}/h3_meta_{args.subset}.jsonl"
LORA = args.lora or "/nfs/danze/model/minimax_h3_vrising/smoke/epoch-4.safetensors"
OUTDIR = args.outdir or f"/nfs/danze/eval/vrising_{args.subset}"

os.makedirs(OUTDIR, exist_ok=True)
rows = [json.loads(l) for l in open(META)]
row = rows[args.index]
clip = os.path.join(DATA_BASE, row["video"])
print(f"clip     : {row['video']}  ({row['category']})")
print(f"prompt   : {row['prompt'][:110]}...")

# 首尾帧取自真实 clip —— 和训练时 train.py 的 input_image/end_image 取法一致
frames = VideoData(clip, height=HEIGHT, width=WIDTH).raw_data()
keyframes = [frames[0], frames[min(NUM_FRAMES, len(frames)) - 1]]
print(f"帧数     : 源 {len(frames)}，取首尾帧做条件")

vram = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
mc = lambda pat: ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern=pat, **vram)
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        mc("FL2VA/text_encoder/model*.safetensors"),
        mc("FL2VA/transformer/model*.safetensors"),
        mc("FL2VA/video_vae/source/model.safetensors"),
        mc("FL2VA/audio_vae/model.safetensors"),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)


def run(tag):
    t0 = time.time()
    video, audio = pipe(
        prompt=row["prompt"],
        height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=args.steps, seed=args.seed,
        keyframes=keyframes, keyframe_indices=[0, -1],
    )
    out = f"{OUTDIR}/{tag}_idx{args.index}_seed{args.seed}.mp4"
    write_video_audio(video=video, audio=audio, output_path=out,
                      fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
    print(f"[{tag}] {out}  帧数={len(video)}  耗时={time.time()-t0:.0f}s  "
          f"峰值显存={torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    return video


print("\n=== base（不挂 LoRA）===", flush=True)
v_base = run("base")

print("\n=== LoRA ===", flush=True)
assert os.path.exists(LORA), f"找不到 {LORA}"
pipe.load_lora(pipe.dit, LORA)
v_lora = run("lora")

# 定量确认两者确实不同：逐帧算平均绝对差
import numpy as np
d = np.mean([np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean()
             for a, b in zip(v_base, v_lora)])
print(f"\nbase vs LoRA 平均像素差: {d:.3f} / 255")
print("→ 接近 0 说明 LoRA 几乎没起作用；明显 >0 说明权重确实生效了")
