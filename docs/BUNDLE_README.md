# MiniMax-H3 微调 — 代码 / 数据 / 文档打包

> 打包于 2026-08-20　base commit `300e3e4da76e`
> 内容涵盖两条数据线：**V-Rising**（已训到 Stage A/B）与 **ABot-World-Explorer-500h**（新，数据就绪）

本 README 只回答一件事：**拿到这个包，怎么把当前进度复现出来。**
设计论证与实测依据在 `docs/` 里，不在这里重复。

---

## 0. 当前进度一览

| 阶段 | ABot 线 | V-Rising 线 |
|---|---|---|
| 数据处理 | ✅ 8000 条切片 + 动作 | ✅ 20699 条 |
| stage 1（latent 缓存） | ✅ 64 条冒烟 | ✅ form_wolf / wolf_life |
| 动作注入 + 落点探针 | ✅ 64/64 PASS | ✅ 1374+64 PASS |
| stage 2（训练） | 🔴 **被 GPU 显存阻塞** | ✅ 已完成 baseline |
| 动作控制端到端验证 | ⛔ **被代码缺口阻塞**（见 §6.1） | ⛔ 同 |

**能立刻复现的**：数据处理全流程、stage 1、动作注入、两个探针。
**需要 GPU 才能继续的**：stage 2（需 ≥55 GB 空闲显存）。

---

## 1. 包内容

```
README.md                              本文
docs/
  minimax_h3_abot_progress.md          ★ ABot 线进展总结（做了什么/怎么做/下一步）
  minimax_h3_abot_data.md              ★ ABot 数据方案（全部实测依据）
  minimax_h3_vrising_finetune.md       V-Rising 数据选型与运维决策
  minimax_h3_world_model.md            动作条件世界模型改造设计
  minimax_h3_architecture_and_data.md  H3 架构、数据契约、帧率重采样
  minimax_h3_vrising_system_overview.md
code/
  abot/abot_action.py                  动作 schema + COLMAP 反推 + 非均匀分箱（带 self-test）
  abot/build_abot_clips.py             切片 + metadata + 动作 npy（带 --verify 对齐抽查）
  abot/inject_abot_action.py           动作注入 stage 1 缓存
  vrising/h3_action.py                 V-Rising 版动作模块
  vrising/inject_action_into_cache.py
  vrising/probe_action_injection.py    ★ 注入落点探针（两条线通用）
  vrising/validate_vrising_lora.py     LoRA A/B 验证
  vrising/build_h3_metadata.py
  vrising/run_h3_all.sh
  scripts/ABot-FL2VA.sh                ABot 训练入口
  scripts/VRising-FL2VA.sh             V-Rising 训练入口
  diffsynth_h3_action.patch            ★ 框架侧改动（196 行，4 个文件）
  diffsynth_base_commit.txt            补丁对应的 base commit
  UNAPPLIED_inference_action_cond.rej  ⚠️ 未应用的 hunk，见 §6.1
data/
  abot_manifest.jsonl                  8000 条切片清单（全局固定顺序）
  abot_meta_{64,2000,8000}.jsonl       训练用 metadata，互为字节前缀
  actions/<prefix>/<sample_id>_w000.npy  8000 个逐帧动作矩阵 [130,17]
```

**不在包内**（体积原因，可复现）：
- `clips/*.mp4` 8000 条切片，38.5 GB —— 用 `build_abot_clips.py` 从源数据重建
- `64-cache/` latent 缓存，822 MB —— 用 stage 1 重建
- 源数据集 2.5 TB，模型权重 135 GB

---

## 2. 环境准备

### 2.1 仓库与补丁

```bash
git clone <DiffSynth-Studio 仓库> DiffSynth-Studio-new
cd DiffSynth-Studio-new
git checkout 300e3e4da76e                      # code/diffsynth_base_commit.txt
git apply /path/to/code/diffsynth_h3_action.patch
cp /path/to/code/scripts/*.sh examples/minimax_h3/model_training/lora/
```

补丁改动 4 个文件，每处的理由见 `docs/minimax_h3_world_model.md` §4：

| 文件 | 改了什么 | 为什么不可省 |
|---|---|---|
| `models/minimax_h3_dit.py` | `enable_action_conditioning()` 延迟建 embedder + forward 注入点 | 建在 `__init__` 里会让 `load_state_dict(strict=True)` 报 Missing key |
| `pipelines/minimax_h3_audio_video.py` | `model_fn_minimax_h3(..., action_cond=None)` **具名形参** | 缓存白名单来自 `inspect.signature`，`**kwargs` 不算数，否则 action 被剪掉 |
| `examples/.../train.py` | `--action_num_buttons` / `--action_train_only` | 开关 |
| `diffusion/runner.py` | stage 1 断点续跑（`.tmp` + `os.replace` 原子写） | 32 小时的作业中途挂掉不能从头再来 |

### 2.2 🔴 PYTHONPATH（不加必然失败）

环境里 `diffsynth` 常是 editable 安装指向**另一个** checkout，而
`accelerate launch` 时 `sys.path[0]` 是脚本目录不是仓库根：

```bash
cd DiffSynth-Studio-new && export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

不设会报 `ModuleNotFoundError: diffsynth.utils.data.minimax_h3`。
两个 `.sh` 里已经带了，手敲命令时要自己加。

### 2.3 模型权重

```bash
export HF_HOME=/nfs/danze/model/.hf_home        # 别落在 home，容易撑爆
huggingface-cli download MiniMaxAI/MiniMax-H3 --local-dir models/MiniMax/MiniMax-H3
```

repo id 是 `MiniMaxAI/MiniMax-H3`（不是 `MiniMax-AI/`）。落盘到
`<repo>/models/MiniMax/MiniMax-H3/`，保持 DiffSynth 默认布局，脚本不用改。
FL2VA 约 135 GB，其中 transformer 62 GB。

> ⚠️ `validate_vrising_lora.py` **必须从仓库根目录跑** —— `local_model_path`
> 相对 cwd 解析，在别处跑会转去 ModelScope 重下（实测 408 kB/s），
> 且它持有的锁会把其他作业一起堵住。

### 2.4 路径改写

代码里的绝对路径集中在这几处，换机器时改：

| 文件 | 常量 |
|---|---|
| `abot/build_abot_clips.py` | `SRC_ROOT` / `OUT_ROOT` |
| `abot/inject_abot_action.py` | `OUT_ROOT` |
| `scripts/ABot-FL2VA.sh` | `DATA_BASE` / `META` / `OUT` |

---

## 3. 复现 ABot 数据处理（不需要 GPU）

### 3.1 自检（秒级，先跑这个）

```bash
python3 code/abot/abot_action.py
```

期望：5 个 `num_frames` 的分组累加全部 OK、抽帧映射有效帧率 ≈24.16、
单帧脉冲落在 token1、yaw 每帧 1 度时 token0=0.250 / token1=1.000（1 帧 vs 4 帧）。

### 3.2 切片

源数据：`acvlab/ABot-World-Explorer-500h`（HuggingFace，30969 条，2.5 TB）。

```bash
python3 code/abot/build_abot_clips.py --plan                      # 先看计划
python3 code/abot/build_abot_clips.py --num-clips 8000 --workers 24 --crf 14 \
        --tiers 64,2000,8000
```

实测：**32 分钟 / 0.24 s 每片 / 零失败 / 38.5 GB**。

产出应与包内 `data/*.jsonl` **逐字节相同** —— 顺序由 `SEED=20260817` 固定，
任何机器可复现。核对：

```bash
md5sum data/abot_meta_8000.jsonl        # 与你重建的对比
```

> ⚠️ 一处例外：`data/*.jsonl` 里第 2280 行的 prompt 经过退化回退处理
> （源 caption 是字面量 `'!!!'`）。这个回退已内置在 `pick_prompt()` 里，
> 所以重建结果仍然一致。

### 3.3 验证对齐（**必跑**）

```bash
python3 code/abot/build_abot_clips.py --verify 10
```

期望 **10/10 PASS**：每条 124 帧、0 重复帧、位移扫描 argmin=0。
输出形如：

```
sample          nfrm  dup  shift-2  shift-1  shift+0  shift+1  shift+2   argmin  verdict
255952735060     124    0   15.86   12.83    6.35   12.22   17.14   +0     PASS
```

判据是 **argmin 落在 0**，不是绝对像素差 —— 参考帧用 PIL bilinear 复现
ffmpeg 的 bicubic，本身有 5~6/255 的地板噪声，慢镜头下和"错一帧"差不多大。
但地板噪声对所有 shift 共同，argmin 仍然干净。

---

## 4. 复现 stage 1（需 GPU ~60 GB）

```bash
cd DiffSynth-Studio-new && export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=<N> STAGE=1 \
  bash examples/minimax_h3/model_training/lora/ABot-FL2VA.sh 64
```

实测：**64 条 / 12 分 41 秒 / 11.9 s 每条 / 822 MB**（含 GPU 争抢）。

### 核对缓存（四项判据）

```bash
python3 - <<'EOF'
import torch
d=torch.load("<...>/64-cache/0/0.pth", map_location="cpu", weights_only=False)
sh,po,_=d
print("latent_t     :", sh['input_latents'].shape[2], "(必须是 37)")
print("audio        :", tuple(sh['audio_input_latents'].shape), "(预期 (2,32,207))")
print("anchor       :", tuple(sh['keyframe_cond_anchor'].shape), "(预期 (780,96))")
print("prompt_embeds:", tuple(po['prompt_embeds'].shape))
print("seq_len      :", po['packed']['seq_len'])
EOF
```

期望：

```
latent_t      37            ← 若是 32 说明 124 帧被静默降级回 107 了
audio         (2, 32, 207)
anchor        (780, 96)
prompt_embeds (964, 5120)
seq_len       16640         = text 964 + anchor 780 + audio 414 + video 14430
```

`latent_t=37` 这一条最重要：它把 `(1,4,4,4,4)` 那套非均匀分组推导与**真实 VAE
输出**对上了，是整条推导链唯一的实证锚点。

> **两个必须带的开关**：`--silent_on_missing_audio`（数据无音轨，缺了直接 crash）、
> 单进程（`data_id` 必须等于 metadata 行号，多进程会变成片内下标）。

---

## 5. 复现动作注入 + 探针（纯 CPU，分钟级）

```bash
python3 code/abot/inject_abot_action.py \
  --meta abot_meta_64.jsonl --cache <...>/64-cache --dry-run    # 先 dry-run
python3 code/abot/inject_abot_action.py \
  --meta abot_meta_64.jsonl --cache <...>/64-cache
```

期望输出：

```
注入 64，跳过(已有) 0，相机通道被置零 40
按键覆盖: W 29  A 28  S 25  D 25  I 20  J 34  K 23  L 40
          Q 0  E 0  Space 0   ← 恒零，天然阴性对照
相机通道 |max|: d_pitch 0.999  d_yaw 1.406  d_roll 0.466
                d_x 3.079  d_y 2.195  d_z 3.870
```

然后跑落点探针（**改了序列布局或 latent_t 后必跑**）：

```bash
python3 code/vrising/probe_action_injection.py --cache <...>/64-cache --n 9999
```

期望 **64/64 通过**，每条形如
`S=16640 v0=2202 latent_t=37 fr=390 n_video=14430 anchor=780`。

探针验六件事，核心是：`narrow`+广播+`cat` 的结果与 `index_add` 参考实现**逐位相同**、
780 行 keyframe anchor 逐行未被触碰、第 k 个 latent token 拿到的确实是 `emb[k]`。

> **为什么必须单独验**：bias 加错位置**训练是验不出来的** —— 加错了 loss 照样降，
> 只是学到的东西没有物理意义。这是会静默失败的问题。

---

## 6. 下一步与已知阻塞

### 6.1 ⛔ 推理时无法传动作（代码缺口）

`code/UNAPPLIED_inference_action_cond.rej` 是一个**静默打失败**的 hunk。现状：

| 位置 | 有 `action_cond` 吗 | 后果 |
|---|---|---|
| `model_fn_minimax_h3`（约 812 行） | ✅ 有 | **训练通路正常**，缓存白名单满足 |
| `MiniMaxH3Pipeline.__call__`（约 99 行） | ❌ **没有** | **推理时传不进动作** |

所以「动作是否真的在控制生成」这个验证**目前做不了** —— 而它正是
`docs/minimax_h3_world_model.md` §7.0d 标为「剩余最高风险」的那一条。

修法是在 `__call__` 签名加 `action_cond: torch.Tensor = None` 并透传到 `model_fn`。
**本包未擅自修改**，因为它超出打包范围且需要跑通验证才算数。

### 6.2 🔴 stage 2 被 GPU 显存阻塞

冒烟 stage 2 试了两次，都因邻居作业膨胀而 OOM。**不是配置问题**：

| 尝试 | 我方峰值 | 崩在哪 | 邻居当时 |
|---|---:|---|---:|
| `--fp8_models dit` | 33.6 GB | 加载权重 | 94 → 106 GB |
| 上面 + 两种 offload | **22.1 GB** | **反向传播** | 106 → **117 GB** |

第二次已跑到 backward，顺带证实 `--enable_model_cpu_offload` 与延迟建立的
`action_embedders` **没有冲突**（曾是个担心点）。

恢复方法 —— 等到有 **≥55 GB 空闲**的卡：

```bash
CUDA_VISIBLE_DEVICES=<N> STAGE=2 ACTION_BUTTONS=17 ACTION_TRAIN_ONLY=1 \
  SAVE_STEPS=50 NUM_EPOCHS=5 \
  bash examples/minimax_h3/model_training/lora/ABot-FL2VA.sh 64

# 卡不够干净时（脚本已支持 EXTRA_ARGS 透传）
EXTRA_ARGS="--fp8_models dit --use_gradient_checkpointing_offload"           # 需 ~40 GB
EXTRA_ARGS="... --enable_model_cpu_offload"                                   # 需 ~25 GB，慢
```

⚠️ **fp8 只适合通路诊断，不适合评估效果**（它改变权重数值）。
诊断本身对 fp8 稳健：`dW = dYᵀ·x`，输入恒零的通道梯度精确为 0，与精度无关。

### 6.3 冒烟 stage 2 要回答什么

**不看 loss 曲线** —— flow matching 每步随机采 timestep，方差压倒性地大，
5000 步只从 0.190 挪到 0.180，没有判据性（`docs/..._vrising_finetune.md` §6.3c）。

判据是存盘权重的**逐通道梯度**（`--action_train_only` 下只含 action_embedders）：

| 问题 | 判据 |
|---|---|
| 17 通道语义有没有串列 | 梯度非零的通道集合必须**精确等于**数据里出现过的通道。Q/E/Space 必须**精确为 0** |
| **有没有捷径学习** | 比按键通道与相机通道的 `\|w\|max`。按键明显偏低 ⇒ 模型只读相机、按键通路没训练 |

第二条是本方案最大的未知数。相机增量是动作的**结果**，信息量远大于按键。
已用 `--cam-dropout 0.5` 预防（把一半样本的相机通道整块置零），但比例是拍的。

### 6.4 正式训练

**2000 条 × 3 epoch = 6000 步**。rank-32 LoRA 63M 参数，1000–10000 步收敛；
同样步数下宁可多数据少 epoch。成本：stage 1 约 6.9 h / 27 GB，stage 2 约 33 h。

因为所有 tier 是字节前缀，**2000 的缓存可被 8000 直接复用**，只需增量编码 6000 条。
两个前提破一条就全废：**单进程**、**只能往后追加不能重排前缀**。

---

## 7. 复现时最容易踩的坑

| 坑 | 症状 | 出处 |
|---|---|---|
| 忘了 `PYTHONPATH` | `ModuleNotFoundError: diffsynth.utils.data.minimax_h3` | §2.2 |
| 忘了 `--silent_on_missing_audio` | KeyError / crash（数据无音轨） | §4 |
| stage 1 用了多进程 | `data_id` 变片内下标，缓存与 metadata 全部错位 | §4 |
| 中途改子集/重排行序 | 已有缓存全部作废（V-Rising 曾因此废掉 83 GB / 20 小时） | §6.4 |
| 拿 loss 曲线判收敛 | 读不出信息，flow matching 方差太大 | §6.3 |
| 改了 `latent_t` 没重跑探针 | bias 落点可能错，且训练不会报错 | §5 |
| `validate_vrising_lora.py` 不在仓库根跑 | 转去 ModelScope 重下并持锁堵住其他作业 | §2.3 |
