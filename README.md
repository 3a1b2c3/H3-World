# Action-Controllable World Model on MiniMax-H3

Turn keyboard input into video: the same first frame plus a different key sequence
should produce a different — and *correct* — motion. Built by fine-tuning
[MiniMax-H3](https://huggingface.co/MiniMax/MiniMax-H3) on
[ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h)
gameplay footage.

The condition path adds **zero trainable parameters**. Key presses become one
short English sentence per latent frame, injected through the text channel the
model was already pretrained on, and bound to the right frame by an attention mask.

## 🎬 Demo

https://github.com/user-attachments/assets/fa6e5961-7ed6-4263-ad04-71c4c9953851

Three chapters, each with the driving action prompt on screen:

1. **Generated vs ground truth** — three held-out clips, each driven by its own recorded keys
2. **Same first frame, eight action prompts** — identical seed and scene, only the keys differ
3. **Longer videos** — one-shot 10.1 s and chunked 15.5 s, with the chunk seams marked

## 📋 Contents

- [How it works](#-how-it-works)
- [Results](#-results)
- [Installation](#️-installation)
- [Quick start](#-quick-start)
- [Training](#-training)
- [Docs](#-docs)

## 🧩 How it works

### Keys → sentence → sequence

Each clip is 124 frames at 24 fps (5.2 s). The video VAE compresses it to **37
latent frames**, so the action track is pooled into 37 steps and each step gets one
sentence from a fixed template:

```
the man <how he moves>, camera <how the view moves>
```

Both slots are a **pure function of 9 bits** — the 8 recorded keys `W A S D I J K L`
plus one synthesized `F` (fast) bit. 60 distinct sentences cover the whole dataset.
Because it is a pure function, training and inference run the exact same code path;
there is no representation to keep in sync.

The 9th bit matters: direction is readable from the keys, **speed is not**. Across
600 clips, `slowly`/`sharply` splits 0.66/0.34 on steps where `J` is held — close to
a coin flip. So `F` is synthesized from the camera rotation rate that COLMAP recovers
from the raw episode, and it is the only information that genuinely exceeds the
original 8 bits.

### Where the sentences go

H3 has no cross-attention; its text already lives in the same self-attention
sequence as the video. So the 37 sentences are simply more text rows:

```
[ text head 566 | 37 sentences 451 | cond 390 | audio 414 | video 37x390 = 14430 | pad 197 ]
                                                                        total 16448
```

Self-attention is permutation-equivariant, so position alone cannot tell the model
which sentence belongs to which frame. A **hard mask** does: sentence *k* and video
frame *k* see each other, sentence *k* and every other video frame do not. Everything
else — video↔video, sentence↔sentence, sentence↔first-frame condition — is untouched,
so removing the sentence rows recovers the original model exactly.

Only a LoRA on `qkv_proj` / `out_proj` is trained (rank 32, 104 modules, 65.6M
parameters). The condition path itself has no new weights.

### Why 37 latents, and why 17

The video VAE is **causal in time** with 4x temporal compression, and encodes in
independent 17-frame clips. Causality is what makes image-to-video work — one image
must map to exactly one latent — and it is why each clip's first output only sees a
single real frame. That gives the frame grouping `1,4,4,4,4` per clip:

```
124 frames -> pad to 136 by repeating the last frame -> 8 clips x 17
           -> each clip encodes independently to 5 latents = 40
           -> token_drop removes the last 3          -> 37 latents
```

Hence `num_frames` must be `17k + 5`: 124 (5.2 s), 243 (10.1 s), 481 (20.0 s).

Full diagrams: [action_injection_arch.html](docs/action_injection_arch.html).

## 📊 Results

**Action control works.** Same first frame, same seed, only the keys change.
Horizontal camera drift measured by phase correlation (sign calibrated on the test
split: negative = camera pans left):

| Action prompt | Drift vs `still` | |
|---|---:|---|
| `camera pans left slowly` | −39 | left ✓ |
| `strafes left` | −12 | left ✓ |
| `stands still` | 0 | baseline |
| `walks forward` | 0 | no pan ✓ |
| `strafes right` | +8 | right ✓ |
| `camera pans right slowly` | +31 | right ✓ |
| `camera pans right sharply` | **+251** | right, **8x stronger** ✓ |

Directions are all correct, the ordering is monotone, and the `F` speed bit produces
an 8x amplitude change. Nothing moves horizontally when the prompt says nothing should.

**Longer videos work without retraining.** Two routes, both measured by frame-to-frame
change (coefficient of variation; lower is steadier):

| Route | Length | CV | Failure mode |
|---|---|---:|---|
| Raise `num_frames` | 10.1 s | 0.944 | no collapse, but hard cuts mid-clip |
| Chunked continuation | 15.5 s | 0.463 | steady, but motion stalls at seams (0.46x) |

Details: [longer_video.md](docs/longer_video.md).

> **What is not yet proven.** These measure *global* responsiveness — that the model
> reacts to the action text. They do not prove *per-frame binding* (that sentence *k*
> controls frame *k*). The criterion for that is "change only sentence *k*, measure
> which frames move", and that tool is not written yet.

## 🛠️ Installation

Tested with CUDA 12.8.

```bash
# 1. environment
conda create -n minimax_h3 python=3.10 -y
conda activate minimax_h3
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. framework — pin the base commit, then apply our patch
git clone https://github.com/modelscope/DiffSynth-Studio.git DiffSynth-Studio-h3
git -C DiffSynth-Studio-h3 checkout $(cat code/diffsynth_base_commit.txt)
git -C DiffSynth-Studio-h3 apply ../code/diffsynth_h3_action.patch

# 3. weights (~135 GB) into DiffSynth-Studio-h3/models/
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('MiniMax/MiniMax-H3',
                  local_dir='DiffSynth-Studio-h3/models/MiniMax/MiniMax-H3')"

# 4. point every cache at a big disk (the root volume is small)
source env.sh
```

`code/diffsynth_h3_action.patch` is the complete framework diff (6 files). It adds
the per-latent text rows, the binding mask, and one fix worth calling out: the
`flex_attention` autotune is disabled, because on H200 one of its candidate kernels
raises an illegal memory access that poisons the whole CUDA context and kills a random
rank at step 0. Autotune bought nothing here (75.6 vs 76.1 ms/iter).

## 🚀 Quick start

### Build the dataset

```bash
# cut clips + per-frame actions from the raw episodes
ABOT_SRC_ROOT=/path/to/ABot-World-Explorer-500h \
  python3 code/abot/build_abot_clips.py --num-clips 20128 --workers 48

# hold out a fixed test split (reuse an existing one when extending the data —
# a hash split reshuffles when the input set grows and leaks old test samples)
python3 code/abot/split_abot_metadata.py \
  --input data/abot_meta_20128.jsonl \
  --train-output data/abot_meta_train_20000.jsonl \
  --test-output  data/abot_meta_test_128_20128.jsonl \
  --fixed-test   data/abot_meta_test_128.jsonl \
  --clips-dir data/clips
```

### Inference

```bash
# 8 GPUs, 8 test samples, builds the review page when it finishes
bash code/abot/run_infer8_text.sh 0 1 2 3 4 5 6 7

# same first frame, one key preset per GPU
bash code/abot/run_action_ab8.sh

# a single sample with an explicit action override
python3 code/abot/infer_abot.py \
  --checkpoint output/minimax_h3_abot/20000_text/step-2500.safetensors \
  --sample-id <sample_id> --device cuda:0 \
  --action-preset pan-right-fast          # or --action-random 0

# longer video: one shot, or chunk-by-chunk
python3 code/abot/infer_abot.py ... --num-frames 243
bash code/abot/run_chunked_continue.sh 3
```

Add `--vram-limit-gib 30 --allow-busy-gpu` to run inference on a GPU that is already
training. Measured cost: **+14.8 GB** and training slows from 9.48 to 14.32 s/it,
recovering the moment inference exits.

### Rebuild the demo reel and review pages

```bash
python3 code/abot/build_demo_video.py      --out demo.mp4   # then drag it into a
                                                           # GitHub comment box to host it
python3 code/abot/build_text_infer_viz.py  --runs output/abot_inference/step9840_text_gpu*
python3 code/abot/build_action_ab_viz.py   --tag ab8_step9840
python3 code/abot/check_page_js.py docs/<page>.html   # run the script under a fake DOM
```

## 🏋️ Training

One command runs the whole gated pipeline — probe, data build, verification, launch.
**Every stage stops on failure**, because this scheme fails silently: a wrong mask
does not raise, it just trains a model that never learns the binding.

```bash
SUBSET=20000 NUM_EPOCHS=100 SAVE_STEPS=2500 \
META=data/abot_meta_train_20000.jsonl \
CACHE=output/minimax_h3_abot/20000_text-cache \
OUT=output/minimax_h3_abot/20000_text \
bash code/scripts/run_text_pipeline.sh
```

| Stage | What it checks |
|---|---|
| 0 · preflight | cache entry count == metadata rows, GPUs idle |
| 1 · mask probe | 6 assertions, including *five consecutive different masks all correct* |
| 2 · data build | 8-way shard; scans the dataset for one uniform padding length |
| 3 · verification | 8 assertions on every entry; `seq_len` must be unique, mirror offset a single negative constant |
| 4 · launch | starts training, waits 3 minutes, confirms it is alive |

Latents must be cached first (`STAGE=1`): ~4.5 h for 20k clips across 8 GPUs,
about 0.17 TB on disk.

Reference run: 7872 clips x 10 epochs = 9840 steps at 9.49 s/it, zero errors.

## 📚 Docs

| Document | What it covers |
|---|---|
| [action_injection_arch.html](docs/action_injection_arch.html) | Architecture diagrams: keys → text, sequence layout, binding mask |
| [journey.md](docs/journey.md) | How this was actually built — two failed approaches, the silent bugs, how the criteria evolved |
| [longer_video.md](docs/longer_video.md) | Generating beyond 5.2 s: three routes and what each costs |
| [pipeline_text_injection.md](docs/pipeline_text_injection.md) | The current scheme, end to end |

## 🙏 Acknowledgements

- [MiniMax-H3](https://huggingface.co/MiniMax/MiniMax-H3) — the base model
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — the training framework
- [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h) — the gameplay data
