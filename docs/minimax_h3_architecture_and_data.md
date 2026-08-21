# MiniMax-H3 架构、数据要求与帧率重采样

> 更新：2026-08-15
> 姊妹文档：[`minimax_h3_vrising_finetune.md`](minimax_h3_vrising_finetune.md)（数据选型、常规微调、运维决策）
> 　　　　　[`minimax_h3_world_model.md`](minimax_h3_world_model.md)（动作条件世界模型改造）
> 　　　　　[`minimax_h3_abot_data.md`](minimax_h3_abot_data.md)（ABot-500h 新数据源的处理方案）
>
> 📌 本文 §5 记的两个硬伤（32.7% 重复帧、尾部丢 0.56s）是 **V-Rising 数据特有的**
> ——源 16fps 上采样到 24fps 才会这样。ABot 数据源是 30fps 下采样，实测 0 重复帧、
> 不丢尾帧，见 [`minimax_h3_abot_data.md`](minimax_h3_abot_data.md) §3.4。

本文回答：**模型长什么样、训练要吃什么、我们喂的东西够不够。**

与两份姊妹文档的分工：那两份记的是「为什么这样做」的决策过程，本文记的是
**底层事实**——架构、接口契约、数据管线的实际行为。事实变得慢，决策变得快。

> 🔴 **本文 §5 含两条此前未记录的发现**（尾部丢帧、重复帧），
> 是 08-15 实测 `LoadVideo` 时发现的，不是从文档推断的。

---

## 0. 速查

| 问题 | 一句话答案 | 详见 |
|---|---|---|
| 什么架构 | 音视频**联合** DiT，单条 packed sequence，50 层**纯 self-attn**（无 cross-attn） | §1 |
| 要什么数据 | video + prompt + input_audio（字段必需）；首尾帧自动取 | §2 |
| 我们的数据够吗 | 视频/prompt ✅；**音轨完全没有** ❌（靠静音兜底）；**帧率有隐性代价** ⚠️ | §3 |
| FL2VA / Ref2VA / T2VA | 「条件 → Video+Audio」的三种条件形态；T2VA 不是独立权重 | §4 |
| 为什么能重采样 | 按时间取**最近邻源帧**，不插值 | §5 |
| 是上采样吗 | 是，72 → 107，靠**帧复制**；且尾部 9 帧被丢弃 | §5.3 |

---

## 1. 训练架构

### 1.1 数据流

```
文本   ──→ text_encoder ──────────────┐
首尾帧 ──→ video_vae ──→ patchify ────┤
音频   ──→ audio_vae  ────────────────┼─→ [ text | cond | audio | video | pad ]
视频   ──→ video_vae ──→ patchify ────┘         单条 [1, S, 5376]
                                                      ↓
                                   50 × DiTBlock(self-attn + MLP)
                                     ↑ AdaLN 按 token_tags 分 3 组调制
                                                      ↓
                                  final_layer ──→ video_logits + audio_logits
```

### 1.2 结构参数

| 项 | 值 |
|---|---|
| `num_layers` | 50 |
| `hidden_size` | 5376 |
| `num_heads` / `head_dim` | 56 / 128 |
| `patch_size` | (1, 2, 2) |
| 训练目标 | flow matching（随机采 timestep 预测速度场） |
| `imgvid_cond_noise_aug` | **0.999**（关键帧 anchor 混入 0.1% 噪声，不是纯净的） |
| `audio_cond_noise_aug` | 1.0 |

### 1.3 pipeline 的 10 个 unit

```
ShapeChecker → NoiseInitializer → InputVideoEmbedder → InputAudioEmbedder
→ VideoRetakeEmbedder → AudioRetakeEmbedder → KeyframeEncoder
→ ReferenceEncoder → PromptEmbedder → PackedSequenceBuilder
```

框架按「是否直接依赖 dit」把这 10 个 unit 切成两组，于是训练天然是两阶段的
（详见姊妹文档 §4）。`VideoRetakeEmbedder` 提供的 `denoise_mask_video`
是 Stage C（history-conditioned rollout）要用的机制。

### 1.4 🔴 最重要的架构事实：没有 cross-attention

50 个 block 里只有 `attn`(self) + `mlp` + `adaln_proj`。**文本不是外挂条件，
而是拼进同一条序列参与 self-attn 的。**

这一条决定了我们能怎么改：

| 想加的条件 | 可行性 |
|---|---|
| 玩家动作 → per-block additive bias | ✅ 照搬 ReactiveGWM 可行，H3 这边甚至更干净（没有 gate 位置要斟酌） |
| NPC 策略 → cross-attn | ❌ **没有现成模块可挂**。应改为「编码成 token 拼进序列」 |

### 1.5 我们加的动作注入（Stage B）

```
action [latent_t, 12] ─→ 第 i 层的 Linear(12→5376, bias=False) ─→ [latent_t, 5376]
                      ─→ 广播加到 video 段的 latent_t × 390 行
```

50 层各一个 embedder，共 `12 × 5376 × 50 = 3,225,600` 参数，**零初始化**
（起点严格等价于原模型）。落点与梯度的验证见世界模型文档 §7.0。

---

## 2. 训练需要什么数据

由 `train.py` 的 `--extra_inputs` 和 `parse_extra_inputs` 决定：

| 字段 | 来源 | 必需性 | 备注 |
|---|---|---|---|
| `video` | metadata | ✅ 必需 | 相对 `--dataset_base_path` |
| `prompt` | metadata | ✅ 必需 | |
| `input_audio` | metadata | ✅ **字段必需** | 缺 key 直接 KeyError；内容可兜底 |
| `input_image` | **自动取 `video[0]`** | 不写进 metadata | 这是选 FL2VA 的直接理由 |
| `end_image` | **自动取 `video[-1]`** | 不写进 metadata | ⚠️ 不是源 clip 的真尾帧，见 §5.3 |
| `action_cond` | 我们加的，走缓存事后注入 | 仅 Stage B | 见世界模型文档 §5 |

`--num_frames` 有硬校验：**必须是 `17n+5`**（video VAE 的时间分组）。

### 2.1 metadata 长什么样

```json
{"video": "bear_to_vampire/clip_000001.mp4",
 "input_audio": "bear_to_vampire/clip_000001.mp4",
 "prompt": "A V-Rising gameplay clip in which the player transforms...",
 "kind": "transform", "category": "transform_bear_to_vampire"}
```

`kind` / `category` 只供我们筛子集，训练不消费。

---

## 3. 我们的数据满足吗

| 要求 | 我们的数据 | 结论 |
|---|---|---|
| 视频 | 20699 条 832×480 | ✅ |
| prompt | 英文，中位 79 词（33–117），已全量标注 | ✅ |
| 首尾帧 | 自动从视频取 | ✅（但见 §5.3 的警告） |
| 文件完整性 | 20699 条逐条 `os.path.exists` 校验，缺失 0 | ✅ |
| **音轨** | **完全没有**（ffprobe 只有一条 h264 video stream） | ❌ **不满足** |
| **帧率 / 帧数** | 81 帧 @16fps，而管线要 24fps | ⚠️ **不满足，代价见 §5** |

### 3.1 音频：不满足，用静音兜底绕过

H3 是音视频联合模型，`input_audio` 不能从 `extra_inputs` 里删。解法：

```
input_audio 指向 mp4 自身 → LoadAudioWithTorchaudio 加载失败返回 None
                          → --silent_on_missing_audio 兜底成静音张量 (2, 142400)
```

**所以命令行必须带 `--silent_on_missing_audio`，否则 crash。**

代价是音频分支占着序列和算力，且模型在学「这个画面对应静音」。但：

```
audio_rows = 178 × 2 = 356      vs      video_rows = 32 × 390 = 12480
```

**不到 3%**。而拆掉它要连带改 `_build_packed_fl2va` 布局、`token_tags`、
AdaLN 索引、`audio_pos`，伤筋动骨。**结论：保持静音兜底，别动。**

### 3.2 帧率：见 §5

---

## 4. 三个变体：FL2VA / Ref2VA / T2VA

命名规律是 `<条件> to <输出>`，**`VA` = Video + Audio**（H3 总是同时生成两者）。

| 变体 | 全称 | 条件 | 调用方式 |
|---|---|---|---|
| **T2VA** | **T**ext → Video+Audio | 只有文本 | `pipe(prompt=...)` |
| **FL2VA** | **F**irst&**L**ast frame → Video+Audio | 文本 + 首帧 + 尾帧 | `pipe(..., keyframes=[first,last], keyframe_indices=[0,-1])` |
| **Ref2VA** | **Ref**erence → Video+Audio | 文本 + 参考视频/音频/图像 | `pipe(..., references=[{"type":"video_audio",...}])` |

### 4.1 三个容易搞混的点

**1. T2VA 不是独立权重。** 官方示例 `MiniMax-H3-FL2VA.py` 用**同一个 pipeline**
先生成 `t2va.mp4`（不传 keyframes）再生成 `fl2va.mp4`（传 keyframes）。
所以 T2VA 是「FL2VA 不给关键帧」的退化用法，不是另一套模型。

**2. Ref2VA 是另一套权重，本地没有。** `models/MiniMax/MiniMax-H3/` 下只有
`FL2VA/`（135 GB）。Ref2VA 还需要额外的 Qwen3-VL processor，且 prompt 格式
完全不同——是结构化的 `subject_definitions` / `retention_analysis` /
`detailed_description` 段落，用于视频编辑、音色迁移这类任务。

**3. 为什么我们选 FL2VA。**

| 变体 | 判断 |
|---|---|
| **FL2VA** | ✅ `transform` 类 clip 天然是「首帧形态 A → 尾帧形态 B」，而 `train.py` 恰好自动取首尾帧。首尾帧是最强的监督信号 |
| Ref2VA | ❌ metadata 里没有参考图，且权重和 processor 都没有 |
| T2VA | ❌ 白白丢掉首尾帧信号 |

### 4.2 关键帧是怎么进模型的

`MiniMaxH3Unit_KeyframeEncoder`：

```
首帧/尾帧 → video_vae.encode_video(process_image=True) → [1,24,1,H',W']
          → patchify_video → 每帧 390 行
          → 2 帧共 780 行，混入噪声（imgvid_cond_noise_aug=0.999）
          → keyframe_cond_anchor，拼在序列的 cond 段
```

> 🔴 这 780 行 anchor 正是 Stage B 的动作 bias **必须避开**的东西——
> 糊上去就污染了首尾帧条件。这也是 `probe_action_injection.py` 专门验的事。

---

## 5. 帧率重采样：不是插值，是复制，而且丢了尾巴

### 5.1 机制

源数据 **81 帧 @16fps = 5.0625 秒**，而 H3 的数据管线写死
`fix_frame_rate=True, frame_rate=24`（`diffsynth/utils/data/minimax_h3.py:51`、
`examples/minimax_h3/model_training/train.py:196`）。

核心是 `core/data/operators.py:166`：

```python
def map_single_frame_id(self, new_sequence_id, raw_frame_rate, total_raw_frames):
    target_time_in_seconds = new_sequence_id / self.frame_rate      # 24
    raw_frame_index_float = target_time_in_seconds * raw_frame_rate # 16
    frame_id = int(round(raw_frame_index_float))
    return min(frame_id, total_raw_frames - 1)
```

**它是「按时间去源视频里取最近邻的那一帧」，没有任何插值。**
对我们的数据就是 `raw = round(j × 2/3)`：

```
输出帧 j:  0  1  2  3  4  5  6  7  8  9 10 11 ...
源帧 raw:  0  1  1  2  3  3  4  5  5  6  7  7 ...
              ↑复制    ↑复制    ↑复制    ↑复制
```

**每 3 个输出帧里有 1 个是重复的。**

所以：是**上采样**（72 个不同源帧 → 107 个输出帧），但靠**帧复制**，
不是生成新内容。

### 5.2 实测验证（不是推算）

```python
from diffsynth.core.data.operators import LoadVideo
lv = LoadVideo(num_frames=107, time_division_factor=17, time_division_remainder=5,
               frame_rate=24, fix_frame_rate=True)
frames = lv("vampire_to_wolf/clip_000001.mp4")
```

```
训练实际拿到的帧数: 107
训练尾帧 == 源第 [71] 帧            ← 不是第 80 帧
源尾帧(80) vs 训练尾帧 平均像素差: 23.75/255
训练序列里与前一帧完全相同的帧: 35/107 = 32.7%
用到的源帧范围: 0 .. 71（源共 81 帧，下标 0..80）
被丢弃的尾部源帧: 9 帧 = 0.562s
输出时长 4.4583s  vs  源时长 5.0625s
```

### 5.3 🔴 后果一：每条 clip 尾部 0.56 秒被静默丢弃

源帧 72–80 **从来没进过训练**。

对 FL2VA 尤其值得注意：`end_image` 拿的是源第 71 帧，**不是 clip 真正的最后一帧**，
两者像素差 **23.75/255**（比 step-5000 LoRA 的整体效果 15.05/255 还大）。

抽了一条 `vampire_to_wolf` 看丢掉的那段（存在 `/nfs/danze/eval/dropped_tail.png`，
源帧 71/74/77/80）：**变身在第 71 帧之前就完成了**，丢掉的主要是变身后的奔跑。
所以「尾帧是形态 B」这个 FL2VA 前提**仍然成立**。

> ⚠️ 但这是**一条 clip 的观察，不是全体的保证**。如果某些类别的变身收尾更慢，
> `end_image` 可能落在变身中途。要下结论需要抽样统计，尚未做。

### 5.4 🔴 后果二：1/3 的帧是静止的 —— 对世界模型更要命

模型看到的是「按着 W，但每三帧里有一帧画面完全不动」。
对一个要学「动作 → 画面变化」因果的世界模型，这是**被写进训练数据的系统性噪声**。

Stage A（学画风）受影响小；**Stage B/C 受影响大**，因为它们要学的正是时序动力学。

> ✅ **动作侧是一致的**：`h3_action.py` 的 `resample_to_video_timeline` 用了
> 逐字相同的公式（`round(j/target_fps*raw_fps)` + clamp），所以动作和画面
> 对齐在同一条（被截断的）时间轴上，**没有额外错位**。

### 5.5 为什么 107 是被逼出来的

`--num_frames` 必须是 `17n+5`，且不能超过可用帧数
（`available = floor(5.0625 × 24) = 121`）：

| `num_frames` | 是否可用 |
|---|---|
| 124 = 17×7+5 | ❌ > 121，`get_num_frames` 会逐 1 递减到满足 `%17==5`，即降级回 107 |
| **107 = 17×6+5** | ✅ 当前值 |

**在 24fps 下 107 是唯一可选值。** 覆盖不到完整 5.06 秒是硬约束的结果，
不是配置失误。

> 📌 顺带澄清姊妹文档 §3 的一处：那里说「124 和 107 都会稳定降级到 107」是对的，
> 但没说清降级后**只覆盖 4.46 秒**。补在这里。

### 5.6 改成 16fps？—— 是真权衡，不该顺手改

| `frame_rate` | `num_frames` | 覆盖时长 | 重复帧 | `latent_t` |
|---|---|---|---|---|
| **24（当前）** | 107 | 4.458s | **35 (32.7%)** | 32 |
| 16（源生） | 73 | 4.5625s | **0** | 22 |

16fps 看起来全面更优（无重复帧、覆盖更长、latent 少 31% 省算力，
因为 `raw = round(j/16*16) = j` 是恒等映射）。**但不建议现在改**：

1. **预训练模型的运动先验是 24fps 的。** 喂 16fps 的帧却当 24fps 用，
   等于告诉模型「每帧的运动量是原来的 1.5 倍」，是分布偏移。
2. **`/24` 在代码里是写死的**，例如 `train.py` 的音频兜底
   `800 * round(len(data["video"]) / 24 * 40)`、pipeline 的 `_FRAME_RESCALE`。
   改帧率会连带打乱音频时间轴。

> 记在这里是为了 Stage C/D 决策时不用重新发现。当前不动。

---

## 6. 复现命令

```bash
cd /nfs/danze/repo/DiffSynth-Studio-new
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# 看源 clip 的真实帧率/帧数
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,r_frame_rate,duration -of default=nw=1 \
  /nfs/danze/data/v_rising/.../wolf/clip_000001.mp4

# 复现 §5.2 的重采样实测
python3 -c "
from diffsynth.core.data.operators import LoadVideo
import numpy as np, imageio
clip='.../vampire_to_wolf/clip_000001.mp4'
lv=LoadVideo(num_frames=107, time_division_factor=17, time_division_remainder=5,
             frame_rate=24, fix_frame_rate=True)
frames=lv(clip); last=np.array(frames[-1])
r=imageio.get_reader(clip); raw=[np.array(r.get_data(i)) for i in range(81)]
print('训练尾帧 == 源第', [i for i,f in enumerate(raw) if np.array_equal(f,last)], '帧')
print('重复帧:', sum(1 for a,b in zip(frames,frames[1:]) if np.array_equal(np.array(a),np.array(b))), '/107')
"
```

推理产出（A/B 对比视频）：

| 路径 | 内容 |
|---|---|
| `/nfs/danze/eval/vrising_form_wolf_step5000/` | step-5000 的 base / lora + 抽帧对比 `*_strip.png` |
| `/nfs/danze/eval/vrising_smoke/` | 08-13 冒烟那轮的 base / lora |
| `/nfs/danze/eval/dropped_tail.png` | §5.3 被丢弃的尾部源帧（71/74/77/80） |

---

## 7. 由本文引出的待决项

1. **`end_image` 落点的抽样统计。** §5.3 只看了一条 clip，变身在丢帧前完成。
   需要跨类别抽样确认这对 `transform_*` 全体成立，否则 FL2VA 的前提有裂缝。
2. **重复帧对 Stage B/C 的实际影响未评估。** 32.7% 的静止帧会不会让模型学出
   「动作有 1/3 概率无效」，需要在带动作的生成结果里观察。
3. **是否为 Stage C/D 换帧率。** §5.6 的权衡在长程 rollout 场景下可能翻转——
   那时时间连续性比匹配预训练先验更重要。

---

## 8. 变更记录

| 日期 | 变更 |
|---|---|
| 08-15 | 初版。§1 架构（含「无 cross-attn」这条关键约束）、§2 数据契约、§3 满足度（音轨缺失 + 帧率）、§4 三变体辨析（T2VA 非独立权重、Ref2VA 权重不在本地）、**§5 帧率重采样的两条新发现**（尾部丢 9 帧 / 32.7% 重复帧，均为 `LoadVideo` 实测）、§6 复现命令与产出位置 |
