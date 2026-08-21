# MiniMax-H3 × V-Rising：模型、数据与实验流程总览

> 更新日期：2026-08-17  
> 项目目标：把 MiniMax-H3-FL2VA 微调为能够根据玩家动作生成后续游戏画面的 V-Rising 世界模型。  
> 配套框图：[`minimax_h3_vrising_architecture.html`](minimax_h3_vrising_architecture.html)

## 0. 一页结论

当前使用的底模是 **MiniMax-H3-FL2VA**，不是 Ref2VA，也不是独立的 T2VA 权重。模型以文本、首帧、尾帧和音频为条件，同时预测视频和音频；V-Rising 数据没有音轨，因此音频分支使用静音占位。

我们在 H3 的 50 个 DiT block 前分别加入一个零初始化的 `Linear(12, 5376, bias=False)`，把 12 维玩家动作变成逐层 additive bias，并且只加到真实 video token，避开首尾帧 anchor、文本、音频和 padding。

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| Stage A | FL2VA LoRA，学习 V-Rising 画风与狼形态先验 | `form_wolf` 1374 条已完成 5 epoch / 6870 步 |
| Stage A 扩展 | 狼形态 + vampire↔wolf 变身 | `wolf_life` 5430 条 Stage 1 cache 已完成，Stage 2 未开始 |
| Stage B | 玩家动作 additive bias | 代码、空间落点、梯度、通道语义已验证；正式训练和时序验证未完成 |
| Stage C | history-conditioned 分段 rollout | 未开始 |
| Stage D | causal attention + KV cache | 未开始 |
| Turbo | 4/8 NFE 少步推理 | 尚未接入；Turbo 本身不包含玩家动作注入 |

最准确的项目定位是：**普通 LoRA baseline 已完成，动作通路已跑通，正式动作条件训练尚未开始。**

---

## 1. 当前使用的模型

### 1.1 模型选择

底模：`MiniMax-H3-FL2VA`

本地权重：

```text
/nfs/danze/repo/DiffSynth-Studio-new/models/MiniMax/MiniMax-H3/FL2VA/
```

FL2VA 是 **First & Last Frame to Video + Audio**：

```text
prompt + 首帧 + 尾帧 + input_audio
                    ↓
            MiniMax-H3-FL2VA
                    ↓
             video + audio
```

选择 FL2VA 的原因是 V-Rising 变身数据天然具有“形态 A 首帧 → 形态 B 尾帧”的结构，首尾帧是比纯文本更强的条件。T2VA 只是同一套 FL2VA 权重在不提供关键帧时的退化用法；Ref2VA 使用另一套 transformer 权重和参考输入管线，当前项目没有使用。

### 1.2 H3 骨干结构

| 项 | 当前值 |
|---|---:|
| DiT block 数 | 50 |
| hidden size | 5376 |
| attention heads | 56 |
| head dim | 128 |
| patch size | `(1, 2, 2)` |
| 训练目标 | flow matching |
| 主干注意力 | 单流 self-attention |
| cross-attention | 无 |
| 输出 | video logits + audio logits |

H3 不是“视频流做 self-attention、文本从 cross-attention 接入”的常见结构。它先把各模态投影到同一个 `5376` 维空间，再构造一条 packed sequence：

```text
[ text | cond(first/last-frame anchor) | audio | video | pad ]
```

50 个 DiT block 都在这条序列上做 self-attention。`token_tags` 用来区分文本、视频和音频模态，并选择对应的 AdaLN 调制。

### 1.3 107 帧配置下的实际形状

当前训练分辨率为 `480×832`，训练序列为 107 帧：

| 张量/区域 | 形状或长度 |
|---|---:|
| video latent | `(1, 24, 32, 30, 52)` |
| latent 时间长度 | 32 |
| 每个 latent 帧的空间行数 | `(30/2)×(52/2)=390` |
| video rows | `32×390=12480` |
| 首尾帧 anchor rows | `2×390=780` |
| audio latent | `(2, 32, 178)` |
| audio rows | 356 |
| text rows | 随 prompt 长度变化 |

音频 rows 不到 video rows 的 3%。尽管世界模型不需要声音，拆除音频分支会破坏 packed layout、token tag、AdaLN 和位置索引，当前选择是保留音频结构并使用静音。

---

## 2. 玩家动作如何注入 H3

### 2.1 动作 schema

动作固定为 12 个二值通道，顺序不能改变：

```text
W, A, S, D, Mouse0, Space, LeftControl, LeftShift,
Q, Alpha1, Alpha2, Alpha3
```

`Alpha1/2/3` 用于消除变身键语义歧义：原始录制中的变身都使用 `LeftControl`，仅凭按键无法知道目标形态，因此按数据类别增加目标通道：

```text
vampire_to_wolf : LeftControl + Alpha1
vampire_to_bear : LeftControl + Alpha2
vampire_to_rat  : LeftControl + Alpha3
```

### 2.2 从 sidecar 到 action_cond

每个 MP4 有一个同名动作 sidecar：

```text
wolf/clip_000001.mp4
wolf/clip_000001.json
```

转换过程：

```text
逐帧 active_keys / click_events
            ↓
     [81, 12] @ 16 fps
            ↓ 与视频完全相同的最近邻帧率映射
    [107, 12] @ 24 fps
            ↓ H3 非均匀时间分组 + window amax
      action_cond [32, 12]
```

H3 video VAE 的每 5 个 latent token 覆盖 17 帧，时间跨度为：

```text
1, 4, 4, 4, 4, 1, 4, 4, 4, 4, ...
```

因此不能使用均匀的 `adaptive_max_pool1d`。当前实现逐段取 `amax`：窗口内只要按键出现过，对应 latent token 就记为按下。

### 2.3 每层 additive bias

加载完预训练 H3 权重后，动态创建 50 个动作投影层：

```python
action_embedders = ModuleList([
    Linear(12, 5376, bias=False) for _ in range(50)
])
```

所有权重零初始化，保证刚接入时：

```text
action bias = 0
带动作分支的模型输出 == 原始 H3 输出
```

第 `i` 个 block 的注入为：

```text
action_cond [32,12]
       ↓ Linear_i(12→5376)
action_embed_i [32,5376]
       ↓ 对每个 latent 帧广播到 390 个空间 token
bias_i [32,390,5376]
       ↓ 只加到 video 区域
DiT block_i
```

公式可写成：

```text
h_video^(i) ← h_video^(i) + W_action^(i) · action_cond
```

关键点是 packed sequence 的 `img_pos` 同时包含关键帧 anchor 和真实视频行。实现使用 `img_pos[cond_rows_count]` 动态定位 video 段起点，只修改连续的 12480 个 video rows：

```text
text       不修改
cond       不修改  ← 避免污染首尾帧 anchor
audio      不修改
video      加动作 bias
pad        不修改
```

动作投影层总参数量：

```text
12 × 5376 × 50 = 3,225,600
```

### 2.4 已完成的动作验证

| 验证 | 结果 |
|---|---|
| action tensor 能否进入 Stage 1 cache | 通过 |
| action bias 是否只落在 video rows | 1374 + 64 条 cache 全通过 |
| 是否避开 780 行关键帧 anchor | 通过 |
| 零初始化是否与原模型等价 | 逐位通过 |
| 前向显存与速度 | 与 baseline 基本一致 |
| 50 层是否都有梯度 | 50/50 非零 |
| 按键列是否串位 | 非零梯度通道与数据中出现的键精确对应 |
| action 与画面响应的时序是否正确 | **尚未通过生成验证** |

因此 Stage B 当前是“实现正确、训练通路正确”，但还不能宣称模型已经学会动作控制。

---

## 3. 训练数据

### 3.1 原始数据

数据根目录：

```text
/nfs/danze/data/v_rising/vrising_train_bundle/data/transform_classified/
```

原始全集共 20699 条，每条 clip 的基本属性为：

| 项 | 值 |
|---|---|
| 视频 | H.264 MP4 |
| 分辨率 | 832×480 |
| 帧数 | 81 |
| 帧率 | 16 fps |
| 时长 | 约 5.06 s |
| 音轨 | 无 |
| prompt | 英文长描述 |
| 动作 | 同名 JSON sidecar，81 帧逐帧记录 |

目录示例：

```text
transform_classified/
├── train.jsonl
├── wolf/
│   ├── clip_000001.mp4
│   └── clip_000001.json
├── vampire_to_wolf/
│   ├── clip_000001.mp4
│   └── clip_000001.json
└── wolf_to_vampire/
    ├── clip_000001.mp4
    └── clip_000001.json
```

### 3.2 H3 metadata

训练入口读取 JSONL，一行一条样本：

```json
{
  "video": "vampire_to_wolf/clip_000001.mp4",
  "input_audio": "vampire_to_wolf/clip_000001.mp4",
  "prompt": "A V-Rising gameplay clip in which the player transforms from vampire into wolf...",
  "kind": "transform",
  "category": "transform_vampire_to_wolf"
}
```

| 字段 | 必需性 | 消费者 |
|---|---|---|
| `video` | 必需 | 数据加载器、video VAE |
| `prompt` | 必需 | text encoder |
| `input_audio` | key 必须存在 | audio loader / 静音兜底 |
| `kind` | 模型不消费 | 子集筛选、分析 |
| `category` | 模型不直接消费 | 子集筛选、动作 remap |

`input_image` 和 `end_image` 不写进 metadata。训练代码从加载后的 `data["video"][0]` 和 `data["video"][-1]` 自动提取首尾帧，作为 FL2VA keyframe condition。

由于视频没有音轨，`input_audio` 仍指向同一个 MP4；加载失败后，`--silent_on_missing_audio` 创建 32 kHz 双声道静音张量。这个 key 不能省略，否则 `parse_extra_inputs` 会直接报错。

### 3.3 动作 sidecar 的两种格式

静态形态数据通常直接是帧数组：

```json
[
  {
    "frame_index": 0,
    "timestamp": 0.0,
    "active_keys": ["W"],
    "click_events": []
  }
]
```

变身和技能数据通常带 metadata：

```json
{
  "metadata": {
    "num_frames": 81,
    "target_fps": 16
  },
  "frames": [
    {
      "frame_index": 0,
      "relative_timestamp": 0.0,
      "active_keys": ["LeftControl"],
      "click_events": []
    }
  ]
}
```

动作解析器必须同时支持这两种结构。

### 3.4 当前子集

| metadata | 数量 | 内容 | 当前用途 |
|---|---:|---|---|
| `h3_meta_smoke.jsonl` | 64 | 固定 seed 随机样本 | 全流程冒烟、动作通路验证 |
| `h3_meta_form_wolf.jsonl` | 1374 | 狼形态活动 | 已完成普通 LoRA baseline |
| `h3_meta_wolf_life.jsonl` | 5430 | form_wolf + 双向变身 | 正式动作实验候选集 |
| `h3_meta_transform.jsonl` | 9314 | 全部变身 | 保留，尚未投入当前实验 |
| `h3_meta_all.jsonl` | 20699 | 全集 | 已停止使用，训练步数过量 |

`wolf_life` 的实际顺序和数量：

```text
0    ... 1373 : form_wolf                    1374
1374 ... 3401 : transform_vampire_to_wolf    2028
3402 ... 5429 : transform_wolf_to_vampire    2028
```

这个前缀顺序允许直接复用 `form_wolf-cache/0/0.pth ... 1373.pth`。

### 3.5 视频重采样的已知问题

H3 要求 `num_frames = 17n+5`，当前使用 `107=17×6+5`。数据加载器把 16 fps 源视频按最近邻方式映射到 24 fps，而不是插值：

```text
目标帧 j → round(j / 24 × 16) → 源帧索引
```

实测后果：

| 问题 | 数值 |
|---|---:|
| 107 帧中的重复帧 | 35，约 32.7% |
| 实际用到的源帧 | 0～71 |
| 丢弃的源尾帧 | 72～80，共 9 帧 |
| 丢失时长 | 约 0.56 s |

动作侧使用完全相同的帧率映射，因此动作与“训练实际看到的视频”一致；但重复帧会给动作动力学带来系统性噪声，这是 Stage B/C 必须通过生成实验评估的风险。

### 3.6 Stage 1 latent cache

训练分两阶段。Stage 1 使用冻结的 text encoder、video VAE、audio VAE 和 keyframe encoder，把每条样本保存为：

```text
<subset>-cache/0/<data_id>.pth
```

`.pth` 的顶层结构为：

```python
(
    inputs_shared,
    inputs_posi,
    inputs_nega,
)
```

典型内容：

```text
inputs_shared:
  input_latents          (1,24,32,30,52)
  audio_input_latents    (2,32,178)
  keyframe_cond_anchor   (780,96)
  action_cond            (32,12)       # 注入后才存在

inputs_posi:
  prompt_embeds          (text_len,5120)
  packed.img_pos / audio_pos / text_pos
  packed.token_tags / cu_seqlens / seq_len
```

Stage 2 直接读取 cache，不再运行三个大编码器。这既节省重复计算，也让 Stage 2 只加载 transformer。

当前 `wolf_life-cache` 已有 5430/5430 条视频 latent，但只有复用的前 1374 条已含 `action_cond`；新增的 4056 条必须先运行 `inject_action_into_cache.py` 才能用于动作训练。

---

## 4. LoRA 与训练配置

Stage A 当前配置：

| 配置 | 值 |
|---|---:|
| learning rate | `1e-4` |
| epochs | 5 |
| LoRA rank | 32 |
| LoRA target | `qkv_proj`, `out_proj` |
| LoRA 参数量 | 约 63M |
| gradient checkpointing | 开启 |
| loss | FlowMatch SFT audio-video loss |

动作训练有两种模式：

1. `ACTION_TRAIN_ONLY=1`：冻结底模和 LoRA，只训练 3.2M action embedder，用于验证动作通路。
2. LoRA + action 联合训练：同时学习 V-Rising 域适配和动作控制，是正式 Stage B 应采用的模式。

`form_wolf` 已完成的 6870 步是**纯 LoRA baseline**，没有传 `ACTION_BUTTONS=12`，因此 cache 中即使存在动作，模型也没有消费。

---

## 5. 实验步骤与结果

### 5.1 已执行步骤

#### 步骤 1：数据审计

- 确认 20699 条视频存在，缺失 0 条。
- 确认视频统一为 832×480、81 帧、16 fps。
- 确认没有音轨，决定使用静音兜底。
- 确认 prompt、类别和同名动作 sidecar 完整。
- 发现并记录 107 帧重采样造成的重复帧与尾部丢帧。

#### 步骤 2：构建 H3 metadata

- 从 `train.jsonl` 保留 `video/prompt/kind/category`。
- 增加必须的 `input_audio` key。
- 构建 smoke、form_wolf、wolf_life、transform、all 子集。
- 保持 `wolf_life` 的 form_wolf 前缀，复用 Stage 1 cache。

#### 步骤 3：64 条 smoke 全流程

- Stage 1 编码成功。
- Stage 2 LoRA 训练成功。
- 同 seed 的 base/LoRA A/B 生成像素差明显非零，确认 LoRA 生效。

#### 步骤 4：调整训练规模

- 原 `all` 计划为 `20699×5=103495` 步，对 rank-32 LoRA 明显过量。
- `all` Stage 1 在 7263/20699 时停止，留下约 83 GB cache。
- 切换到 `form_wolf` 1374 条，将 5 epoch 控制在 6870 步。

#### 步骤 5：完成 form_wolf baseline

- Stage 1：1374/1374 完成。
- Stage 2：5 epoch / 6870 步完成。
- 保存 `step-500 ... step-6870` 共 14 个 checkpoint。
- step-5000 与 base 有明显差异，但目视更软，尚需 checkpoint sweep 选择最佳权重。
- flow-matching loss 方差主要由随机 timestep 决定，不能用作 checkpoint 质量排序。

#### 步骤 6：实现动作数据管线

- 同时解析 list 和 `{metadata, frames}` 两种 sidecar。
- 构建 12 键动作 schema。
- 实现 LeftControl → Alpha1/2/3 语义 remap。
- 复刻视频的 16→24 fps 最近邻映射。
- 按 `(1,4,4,4,4)` 时间跨度生成 `[32,12]` action tensor。
- 事后把 action tensor 原子写入 Stage 1 cache。

#### 步骤 7：实现和验证动作注入

- 在 50 个 block 前加入独立零初始化动作投影。
- 动态计算 video span，跳过关键帧 anchor。
- CPU 探针在 1374 + 64 条真实 packed metadata 上全部通过。
- smoke action-only 训练完成 320 步。
- 50/50 层持续获得梯度，动作通道与数据按键覆盖严格一致。

#### 步骤 8：扩展 wolf_life cache

- 复用前 1374 条 `form_wolf` cache。
- 只编码新增 4056 条双向变身数据。
- `wolf_life` Stage 1 最终完成 5430/5430。
- 新增 4056 条当前还没有 `action_cond`。

### 5.2 下一轮实验

建议按以下顺序推进：

1. 对 `wolf_life-cache` 注入全部动作，抽检 0、1373、1374、5429 四个边界样本。
2. 重新运行 action injection probe，确保 5430 条的 video span 全部正确。
3. 对 `form_wolf` 做 checkpoint sweep：至少比较 step-500、2000、5000、6870。
4. 使用同 seed、同首尾帧、同 prompt 比较 base/各 checkpoint，并加入 GT 中间帧指标。
5. 选择 Stage A 权重后，启动 LoRA + action 联合训练。
6. 构造动作对照：相同 prompt/seed/keyframes，仅改变 W/A/S/D/变身通道。
7. 检查动作开始和画面响应的相对时间，关闭 Stage B 最后的时序风险。
8. 动作控制成立后，再实现 `denoise_mask_video` 的 history-conditioned rollout。

### 5.3 完成 Stage B 的判据

Stage B 只有同时满足以下条件才算完成：

- 相同视觉条件下，不同动作能稳定产生方向或变身差异；
- 无动作时模型不会凭空运动；
- 动作响应发生在正确 latent 时间段；
- 关键帧约束没有被动作 bias 污染；
- 连续多个 clip 的动作效果不快速衰减；
- 结果不是仅靠 prompt 或尾帧泄漏得到。

---

## 6. Turbo 是什么，以及如何与动作模型组合

### 6.1 Turbo 的含义

ModelTC/LightX2V 的 MiniMax-H3-Turbo 是作用在官方 H3 transformer 上的**少步蒸馏 LoRA**：

```text
原始 H3：约 50 NFE
Turbo：  4 或 8 NFE
```

NFE 是完整 transformer 的调用次数。Turbo 没有减少 H3 的 50 层，也不是独立小模型；它让相同骨干在一次采样中只做 4/8 次大的 flow 更新。

Turbo 仓库目前提供 FL2VA/T2VA、I2VA 和 Ref2VA 的推理 checkpoint，但**没有**：

- `action_cond` 输入；
- 键鼠动作 schema；
- action embedder；
- 动作与 latent 时间轴对齐逻辑；
- 动作层训练权重。

来源：

- <https://github.com/ModelTC/Minimax-H3-Turbo>
- <https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/DIFFUSERS_SETUP_AND_INFERENCE.md>

### 6.2 候选组合

逻辑上的候选模型是：

```text
官方 MiniMax-H3-FL2VA
  ├─ V-Rising LoRA           学画风/形态
  ├─ action_embedders        学玩家控制
  └─ Turbo LoRA              学 4/8 NFE 少步采样
```

每一次 Turbo NFE 内仍执行完整的 50 层 DiT，并在每层执行 action bias：

```text
for denoise_step in 4_or_8_steps:
    for block in 50_blocks:
        video_hidden += action_embedder[block](action_cond)
        video_hidden = block(video_hidden)
```

### 6.3 风险与推荐路线

直接叠加三个适配器不能假定有效：

- V-Rising LoRA 和 Turbo LoRA 都修改 transformer 投影，可能相互干扰；
- 当前动作模型按原始 flow timestep 分布训练；
- Turbo 使用专门的 4/8-step sigma schedule；
- 少步大跳可能改变动作响应强度和发生时间；
- Turbo 仓库使用 Diffusers 权重命名，当前训练使用 DiffSynth，实现间需要做 key 映射和输出一致性检查。

推荐路线：

```text
先完成原始 H3 上的动作条件模型
            ↓
验证控制与时序正确
            ↓
测试叠加 Turbo LoRA 的兼容性
            ↓ 若不稳定
对最终动作模型重新做 4/8-step 蒸馏
```

Turbo 主要加速推理和 rollout，不会让当前 flow-matching 微调按 `50/4` 的比例提速，因为训练每个 batch 本来只采一个随机 timestep 做一次前向/反向。

---

## 7. 关键文件

| 文件 | 作用 |
|---|---|
| `/nfs/danze/data/v_rising/build_h3_metadata.py` | 构建 H3 JSONL 子集 |
| `/nfs/danze/h3_action.py` | sidecar → `[32,12]` 动作张量 |
| `/nfs/danze/inject_action_into_cache.py` | 把动作写入 Stage 1 cache |
| `/nfs/danze/probe_action_injection.py` | 验证动作落点、anchor 隔离和零初始化 |
| `diffsynth/models/minimax_h3_dit.py` | 50 层 action embedder 和 forward 注入 |
| `diffsynth/pipelines/minimax_h3_audio_video.py` | action 参数透传、video span 定位 |
| `examples/minimax_h3/model_training/train.py` | 动作训练开关和两阶段任务 |
| `examples/minimax_h3/model_training/lora/VRising-FL2VA.sh` | 当前训练入口与 LoRA 参数 |
| `/nfs/danze/minimax_h3_vrising_architecture.html` | 可视化框图 |

已有的详细决策记录仍保留在：

- `minimax_h3_architecture_and_data.md`
- `minimax_h3_vrising_finetune.md`
- `minimax_h3_world_model.md`

