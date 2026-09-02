# First Interactive World Model on MiniMax-H3

Turn keyboard input into video: the same first frame plus a different key sequence
should produce a different -- and *correct* -- motion. Built by fine-tuning
[MiniMax-H3](https://huggingface.co/MiniMax/MiniMax-H3) on
[ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h)
gameplay footage.

<a href="https://huggingface.co/DANNY621/H3-World-results"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace Model"></a>
<a href="https://danzer1xxxxchan.github.io/H3-World"><img src="https://img.shields.io/badge/Web-Project Page-1d72b8.svg" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2609.01560">
  <img src="https://img.shields.io/badge/arXiv-H3--World-A42C25.svg" alt="arXiv">
</a>




## Demo

https://github.com/user-attachments/assets/cc2b1099-cc67-407f-8824-e4cddfb5f313


## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training](#training)

## How it works

### Keys -> sentence -> sequence

Each clip is 124 frames at 24 fps (5.2 s). The video VAE compresses it to **37
latent frames**, so the action track is pooled into 37 steps and each step gets one
sentence from a fixed template:

```
the man <how he moves>, camera <how the view moves>
```

Both slots are a **pure function of 9 bits** -- the 8 recorded keys `W A S D I J K L`
plus one synthesized `F` (fast) bit. The action dictionary is 9 character clauses x
16 camera clauses = 144 structurally valid combinations, of which 135 are reachable
and 83 are actually observed across the 7,872-clip training set. Because the
mapping is a pure function, training and inference run the exact same code path;
there is no learned representation to keep in sync.

The 9th bit matters: direction is readable from the keys, **speed is not**. On
steps where `J` is held, `slowly`/`sharply` split close to a coin flip. So `F` is
synthesized from the camera rotation rate recovered from COLMAP pose estimates on
the raw episode -- it's the one channel that genuinely adds information beyond the
raw 8 keys.

### Where the sentences go, and why a directed mask

H3 has no cross-attention; its text already lives in the same self-attention
sequence as the video. So the 37 sentences are simply more text rows, placed at a
fixed offset before their bound video frame.

Self-attention is permutation-equivariant, so position alone cannot tell the model
which sentence belongs to which frame. A mask does. The mask here is **directed**,
not symmetric: annotation row *k* can be *read by* video frame *k* and by itself,
but *cannot itself read* any other annotation row. This closes a bypass that a
symmetric mask leaves open -- with a symmetric mask, annotation *j* can flow into
annotation *k* through their mutual visibility, so video frame *k* reading
annotation *k* ends up reading annotation *j* too, and the guarantee that action
*k* only reaches frame *k* breaks. Everything else -- video<->video, annotation<->
first-frame condition, annotation<->itself -- is untouched, so removing the
annotation rows recovers the original model exactly.

Only a LoRA on `qkv_proj`/`out_proj` is trained: rank 32 across the 50 DiT blocks
plus 2 token-refiner layers (52 layers x 2 modules = 104 modules, 208 tensors,
65.6M parameters, 0.198% of the 33.1B DiT). The condition path itself adds no new
weights.

### Why 37 latents, and why 124 frames

The video VAE is **causal in time** with 4x temporal compression, and encodes in
independent 17-frame clips. That gives the frame grouping `1,4,4,4,4` per clip:

```
124 frames -> 8 clips x 17 (padded from 122)
           -> each clip encodes independently to 5 latents = 40
           -> token_drop removes the last 3          -> 37 latents
```

Hence `num_frames` must be `17k + 5`: 124 (5.2 s), 243 (10.1 s), 481 (20.0 s).

## Results

**The directed mask and adaptation are both necessary, and the mask alone isn't
enough without adaptation.** The test below uses a single instruction whose meaning
changes mid-clip -- the camera pans left for the first 15 latents, then reverses to
pan right for the remaining 22 -- so a correct response has to bind each half of
the instruction to the right half of the video, which a single global sentence
cannot express positionally. Response is dense optical flow (Farneback), summed
separately before and after the switch; positive values mean the camera pans left.

| Condition | Before switch (want >0) | After switch (want <0) |
|---|---:|---:|
| Base H3 + one global instruction, no LoRA | +0.0 | -17.3 |
| Base H3 + per-latent prompts, zero-initialized LoRA | -0.1 | 0.0 |
| **This model** | **+52.7** | **-106.0** |

The zero-LoRA per-latent row shows the base model does not act on latent-aligned
sentences in this input format at all (mean absolute flow 0.003, the clip is
essentially frozen), even though it *does* respond to the same instruction phrased
as one global sentence. Training the LoRA restores directional control and gives
roughly 5x the response magnitude of the global-prompt baseline. Reversing the
instruction order (pan right, then left) gives the same pattern.

## Installation

Tested with CUDA 12.8.

```bash
# 1. environment
conda create -n minimax_h3 python=3.10 -y
conda activate minimax_h3
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. framework -- pin the base commit, then apply this repo's patch
git clone https://github.com/modelscope/DiffSynth-Studio.git DiffSynth-Studio-h3-v2
git -C DiffSynth-Studio-h3-v2 checkout $(cat code/diffsynth_base_commit.txt)
git -C DiffSynth-Studio-h3-v2 apply ../code/diffsynth_h3_action.patch

# 3. weights (~135 GB) into DiffSynth-Studio-h3-v2/models/
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('MiniMax/MiniMax-H3',
                  local_dir='DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3')"

# 4. point every cache at this repo instead of the default locations
source env.sh
```

`code/diffsynth_h3_action.patch` is the complete framework diff (7 files: the
directed attention mask, the per-latent text injection, the packed-sequence
layout, and one training entry point). It also disables `flex_attention`'s
autotune, because on H200 one of its candidate kernels raises an illegal memory
access that poisons the whole CUDA context and kills a random rank at step 0 --
autotune bought nothing here (75.6 vs 76.1 ms/iter against the default heuristic).

**Do not `pip install -e` this checkout.** An editable install registers a finder
that resolves `import diffsynth` back to whatever package it was installed from,
regardless of `sys.path` order or `PYTHONPATH`. If that finder is active, training
and inference silently run against the wrong code with no error and a
normal-looking loss curve -- `code/train.sh` and `code/abot/infer.py` both strip it
before importing diffsynth and hard-fail if the check doesn't pass, but only for
imports that go through them.

## Quick start

### Build the dataset

```bash
# cut clips + per-frame actions from the raw episodes
ABOT_SRC_ROOT=/path/to/ABot-World-Explorer-500h \
  python3 code/abot/build_abot_clips.py --num-clips 8000 --workers 48

# hold out a fixed test split
python3 code/abot/split_abot_metadata.py \
  --input data/abot_meta_8000.jsonl \
  --train-output data/abot_meta_train_7872.jsonl \
  --test-output  data/abot_meta_test_128.jsonl \
  --clips-dir data/clips
```

### Cache latents, then inject the action text

Stage-1 latent caching uses DiffSynth's own dataset builder
(`examples/minimax_h3/model_training/train_v2.py ... --task sft:data_process`,
same invocation as training in `code/train.sh` but with that one flag changed).
Once the cache exists, write the per-latent action sentences into it:

```bash
python3 code/abot/inject_abot_text.py \
  --meta data/abot_meta_train_7872.jsonl \
  --cache output/minimax_h3_abot/7872-cache \
  --device cuda:0
```

### Inference

```bash
python3 code/abot/infer.py \
  --checkpoint output/minimax_h3_abot/7872_directed/step-10000.safetensors \
  --first-frame path/to/first_frame.png \
  --scene-prompt "The scene is ..." \
  --action-preset pan-right-fast \
  --out generated.mp4
```

`--action-preset` covers the basic single-combination vocabulary (`still`,
`forward`, `back`, `strafe-left`, `strafe-right`, `tilt-up`, `tilt-down`,
`pan-left`, `pan-right`, `pan-left-fast`, `pan-right-fast`); the full key ->
sentence rule table, including compound actions, is in
[`code/abot/action_script.py`](code/abot/action_script.py).

## Training

```bash
bash code/train.sh                    # 4 GPUs (0-3) by default
CUDA_VISIBLE_DEVICES=4,5,6,7 bash code/train.sh
```

This is the released configuration: 7,872 training clips, rank-32 LoRA on
`qkv_proj`/`out_proj`, 20 epochs, checkpoints saved every 2,000 steps. Two things
must both point at the patched framework, or training silently uses a different
attention mask against the same data: the script `cd`s into
`DiffSynth-Studio-h3-v2`, and its entry point (`train_v2.py`) strips the
editable-install finder before importing diffsynth and runs three hard
preflight checks -- see "Installation" above for why that finder is a problem.

## Acknowledgements

- [MiniMax-H3](https://huggingface.co/MiniMax/MiniMax-H3) -- the base model
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) -- the training framework
- [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h) -- the gameplay data
