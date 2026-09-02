# First Interactive World Model on MiniMax-H3

https://github.com/user-attachments/assets/1c862995-8809-447e-bade-2c47bfdb2738

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

> **H3-World: Turning Language Understanding into World Control**    
> Danze Chen<sup>12</sup>, Zeqing Wang<sup>12</sup>, Ziyue Lin<sup>3</sup>, [Xingyi Yang](https://adamdad.github.io/)<sup>3</sup> , [Yeying Jin](https://jinyeying.github.io/)<sup>12</sup>  
> <sup>1</sup> Tencent, <sup>2</sup> National University of Singapore, <sup>3</sup> The Hong Kong Polytechnic University


## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Training](#training)

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

## 🔗 Citation
```
@misc{chen2026h3worldturninglanguageunderstanding,
      title={H3-World: Turning Language Understanding into World Control}, 
      author={Danze Chen and Zeqing Wang and Ziyue Lin and Xingyi Yang and Yeying Jin},
      year={2026},
      eprint={2609.01560},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.01560}, 
}
```
