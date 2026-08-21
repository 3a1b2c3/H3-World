# MiniMax-H3 × V-Rising：方法与决策记录

> 更新：2026-08-15
> 代码：`/nfs/danze/repo/DiffSynth-Studio-new`　数据：`/nfs/danze/data/v_rising`
> 姊妹文档：[`minimax_h3_world_model.md`](minimax_h3_world_model.md)（动作条件世界模型改造）
> 　　　　　[`minimax_h3_architecture_and_data.md`](minimax_h3_architecture_and_data.md)（架构、数据契约、帧率重采样）
> 　　　　　[`minimax_h3_abot_data.md`](minimax_h3_abot_data.md)（ABot-500h 新数据源的处理方案）

本文回答一个问题：**为什么是现在这套做法。**
每个决策都写清备选方案、依据、以及依据是实测还是推断。

---

## 0. 目标与当前状态

最终目标是把 H3 做成 V-Rising 的**动作条件游戏世界模型**。分四阶段推进：

| 阶段 | 内容 | 状态 |
|---|---|---|
| **A** | FL2VA LoRA，学画风 + 形态先验 | 🔄 **stage 2 进行中**（`form_wolf`，纯 LoRA baseline） |
| **B** | 加动作 additive bias，chunk 级控制 | ✅ **已跑通并验证**，只剩时序对齐待验（世界模型文档 §7.0） |
| **C** | `denoise_mask_video` 做 history-conditioned rollout | 待定 |
| **D** | causal 化 + KV cache，实时交互 | 远期 |

**进度**（2026-08-15 08:45）：

| 项 | 状态 |
|---|---|
| Stage A `all`(20699) | ❌ **已终止于 7263/20699**（跑了 20h），`all-cache/` 83 GB 成沉没成本。原因见 §7 |
| Stage A `form_wolf`(1374) stage 1 | ✅ 完成，08-14 02:41→06:35（3h55m），16 GB |
| Stage A `form_wolf` stage 2 | 🔄 **5915/6870 步**（epoch 5/5），争抢解除后 **7.93 s/it**，ETA **~10:51** |
| 动作张量注入缓存 | ✅ 完成并实测校验（见世界模型文档 §5） |
| **Stage B 运行时验证** | ✅ **08-15 完成**：注入落点、开销、梯度与通道语义全部验过，见世界模型文档 §7.0 |
| **smoke + action 验证轮** | ✅ **已跑完 5 epoch / 320 步**，权重单调增长到 2.16e-02，50/50 层始终非零 |
| **`wolf_life`(5430) stage 1** | 🔄 **08-15 07:12 启动**，复用 `form_wolf` 的 1374 条缓存，只编码新增 4056 条，**5.21 s/it**，ETA **~13:45** |
| **GPU 争抢** | ✅ **08-15 ~08:25 解除**，速度翻倍，并暴露出原「干净卡」基准是脏的，见 §6.7 |

已完成并验证：smoke（64 条）全流程跑通，LoRA 经 A/B 对比确认生效。

> ⚠️ `form_wolf` 这轮 stage 2 **没带 `--action_num_buttons`**，是纯 LoRA baseline——
> `action_cond` 躺在缓存里不被消费（`num_action_buttons=0` 时注入分支整个跳过）。
> 动作通路的验证没有等它，而是 08-15 在 smoke 上并行做掉了，见 §9.2。

> ✅ **GPU 争抢已于 08-15 ~08:25 解除**（§6.7）。此前实测慢 2.0–2.1×
> ——注意这个数比文档一直写的 1.85× 更大，因为原基准本身就是脏的。

---

## 1. 数据：为什么用 `transform_classified`

`/nfs/danze/data/v_rising` 下有多份产物，只有一份适合 H3：

| 路径 | 判断 |
|---|---|
| `vrising_train_bundle/data/transform_classified/` | ✅ **唯一数据源**，20699 条 clip + 英文长 prompt |
| `vrising_train_bundle/data/processed_data/` | ❌ 给 ReactiveGWM 动作条件模型导出的，且 `source` 指向另一台机器（`/opt/dlami/nvme/...`），实际 `videos/`、`actions/` 不在本机 |
| `ctrl_segments/` | ❌ 仅 79 条且无 prompt |
| `vampire/` | ❌ 是代码和 checkpoint，不是数据 |

### 数据属性（全部实测核对）

| 项 | 值 |
|---|---|
| 分辨率 / 帧率 | 832×480，81 帧 @16fps ≈ 5.06 s |
| 音轨 | **无**（ffprobe 只有一条 h264 video stream） |
| clip 总数 | 20699（另 2 条缺标注被排除） |
| prompt | 英文，中位 79 词（33–117），已全量标注 |
| 文件缺失 | **0**（20699 条 metadata 逐条 `os.path.exists` 校验） |

分布：`transform` 9314 / `static` 8172 / `ability` 3213。

### 子集选择：先选了 `all`，事后推翻，08-14 已切到 `form_wolf`

`transform` 子集（9314 条）在语义上最贴 FL2VA——首帧形态 A、尾帧形态 B。当时选 `all` 的理由：

- Stage A 的目的是学**画风 + 各形态外观**这个底座，static 和 ability 的 8172+3213 条正是形态外观的主要来源
- `transform` 里正反向是**倒放配对**的（`vampire_to_wolf` 2028 ↔ `wolf_to_vampire` 2028），信息冗余明显，有效样本远少于 9314

> 🔴 **这个决策事后被推翻，并已于 08-14 02:41 执行切换。** 上面的推理只考虑了"数据覆盖度"，
> 漏算了"LoRA 需要多少步"——`all` × 5 epoch = 103,495 步，而 63M 参数的 rank-32 LoRA
> 在 1000–10000 步就收敛。详细论证见 **§7**，已切到 `form_wolf`(1374)。
> **代价：`all` 的 stage 1 跑到 7263/20699 被终止，83 GB 缓存因行号索引不可复用（§7.4）。**

> ⚠️ 另一条由倒放配对引出的评估纪律：**切验证集必须按 clip id 分组**，不能按行随机切，
> 否则同一段素材的正放/倒放会分别落进训练和验证集，指标虚高。

---

## 2. 数据格式：三个坑决定了 metadata 长什么样

H3 训练读 `--dataset_metadata_path`，字段由 `train.py` 的 `parse_extra_inputs` 消费。踩了三个坑：

### 坑 1：`input_audio` 是必需字段，但我们的 clip 没音轨

`parse_extra_inputs` 里非 `input_image`/`end_image` 的 key 直接 `data[extra_input]` 取值，缺了就 **KeyError**。而 H3 是音视频联合模型，`input_audio` 不能从 `extra_inputs` 删掉。

**解法**：让 `input_audio` 指向 mp4 自身 → `LoadAudioWithTorchaudio` 加载失败返回 `None` → `--silent_on_missing_audio` 兜底成静音张量。

```python
inputs_shared["input_audio"] = (torch.zeros((2, 800 * round(len(data["video"]) / 24 * 40))), 32000)
```

所以命令行**必须带 `--silent_on_missing_audio`**，否则 crash。实测兜底张量 `(2, 142400)`，正常。

### 坑 2：`input_image` / `end_image` 不写进 metadata

它们由 `train.py` 自动从训练视频取首尾帧，这也正是**选 FL2VA 的理由**：

```python
if extra_input == "input_image":   keyframes.append(data["video"][0]);  keyframe_indices.append(0)
elif extra_input == "end_image":   keyframes.append(data["video"][-1]); keyframe_indices.append(-1)
```

对比三个变体：

| 变体 | 条件 | 判断 |
|---|---|---|
| **FL2VA** | 首帧 + 尾帧 | ✅ 与数据天然对齐，且首尾帧是最强的监督信号 |
| Ref2VA | 参考图（需 Qwen3-VL processor + `references` 字段） | ❌ metadata 里没有参考图 |
| T2VA | 纯文本 | ❌ 白白丢掉首尾帧信号 |

### 坑 3：`--num_frames` 必须是 `17n+5`

`train.py` 入口硬校验（video VAE 的时间分组）。

### 产出的 metadata

`build_h3_metadata.py` 加 `input_audio` 字段并按 kind 拆子集：

| 文件 | 条数 | 用途 |
|---|---|---|
| `h3_meta_smoke.jsonl` | 64 | 冒烟（seed=0 采样） |
| `h3_meta_form_wolf.jsonl` | 1374 | **当前在训** |
| `h3_meta_transform.jsonl` | 9314 | 只留 transform |
| `h3_meta_all.jsonl` | 20699 | 已弃用（§7） |

```json
{"video": "bear_to_vampire/clip_000001.mp4",
 "input_audio": "bear_to_vampire/clip_000001.mp4",
 "prompt": "A V-Rising gameplay clip in which the player transforms from bear into vampire...",
 "kind": "transform", "category": "transform_bear_to_vampire"}
```

`kind`/`category` 仅供我们筛选，训练不消费。

---

## 3. 为什么 `num_frames=107`

`107 = 17×6+5`，是满足 `17n+5` 且不超过可用帧数的最大值：原 clip 81 帧 @16fps，`fix_frame_rate=True` 重采样到 24fps 后只剩 `floor(5.0625×24)=121` 帧。

**实测澄清**：124 和 107 都会**稳定**降级到 107（所有 clip 都是 81 帧，降级确定性），而且 `get_pipeline_inputs` 用的是 `len(data["video"])` 而非 `args.num_frames`，pipeline 会自校正。所以脚本早期注释里"124 会导致 batch 内帧数不一致"的理由**不成立**——但 107 仍是正确取值（显式、满足约束）。

> 🔴 **08-15 补充：107 帧只覆盖源 clip 的 4.46 秒，不是全部 5.06 秒。**
> `fix_frame_rate=True` 是「按时间取最近邻源帧」而非插值，实测结果：
> **每条 clip 尾部 9 帧（0.56s）被静默丢弃**，且 **107 帧里 35 帧（32.7%）
> 是前一帧的逐位复制**。`end_image` 拿到的是源第 71 帧而非第 80 帧。
> 完整分析、实测方法与权衡见
> [`minimax_h3_architecture_and_data.md`](minimax_h3_architecture_and_data.md) §5。

### 由此确定的全部形状（已由真实 cache 交叉验证）

| 量 | 公式 | 值 | 缓存实测 |
|---|---|---|---|
| `latent_t` | `((107-5)//17)*5+2` | 32 | `input_latents=(1,24,**32**,30,52)` ✓ |
| `latent_h,w` | `480//16, 832//16` | 30, 52 | ✓ |
| `frame_rows` | `(30//2)*(52//2)` | 390 | `keyframe_cond_anchor=(**780**,96)=2×390` ✓ |
| `audio_t` | `round(107/24*40)` | 178 | `audio_input_latents=(2,32,**178**)` ✓ |

手算与实测完全吻合，这也让[世界模型文档](minimax_h3_world_model.md)里基于同一套推导的设计有了实证支撑。

---

## 4. 为什么是两阶段

**不是我们的设计，是框架按数据流图自动切的**，但理由充分。

`split_pipeline_units`（`diffusion/base_pipeline.py:485`）找出直接依赖 `dit` 的 unit，沿依赖边做传递闭包，把 pipeline 的 10 个 unit 切成两组：

```python
if task.endswith(":data_process"):  other_units, pipe.units = pipe.split_pipeline_units([...])  # 留预处理
elif task.endswith(":train"):       pipe.units, _ = pipe.split_pipeline_units([...])            # 留 DiT 侧
```

| | Stage 1 `sft:data_process` | Stage 2 `sft:train` |
|---|---|---|
| 加载 | text_encoder + video_vae + audio_vae | transformer |
| 跑的 unit | InputVideo/AudioEmbedder、PromptEmbedder、KeyframeEncoder、PackedSequenceBuilder | DiT 前向 + FlowMatch loss |
| "loss" | `lambda pipe, *args: args`（恒等，只为把结果存盘） | 真 loss，只有 LoRA 拿梯度 |
| 产出 | `{data_id}.pth` = `(inputs_shared, inputs_posi, inputs_nega)` | LoRA 权重 |

### 收益（按实测速率）

编码器是冻结的，同一条 clip 每个 epoch 的 latent 完全相同。

```
融合:   5 × (编码 32h + 训练 52h) = 420h
两阶段: 编码 32h(一次) + 5 × 52h  = 292h      → 省 128h ≈ 5.3 天
```

**次要收益是显存**：四个模型合计 144 GB，而 H200 是 143.77 GB——还没算激活值就装不下。拆开后 stage 2 根本不加载编码器。实测 stage 1 稳定 59.5 GB。

### 代价

1. **磁盘**：11.7 MB/clip × 20699 ≈ **236 GB**
2. **数据增强被冻死**：latent 算一次固定，做不了 per-epoch 随机增强（对我们无影响，本来就没做）
3. **缓存白名单**：stage 1 末尾 `GeneralUnit_RemoveCache` 会剪掉不在白名单的 key，白名单 = `inspect.signature(pipe.model_fn).parameters` + loss 所需参数 —— **这条是 Stage B 的关键约束**，详见世界模型文档 §3

---

## 5. 环境与运维决策

### 5.1 🔴 `PYTHONPATH` —— 不加必然失败

环境里 `diffsynth` 是 **editable 安装，指向 `/data/danzechen/DiffSynth-Studio`**（另一个旧 checkout）。而 `accelerate launch examples/.../train.py` 时 `sys.path[0]` 是**脚本所在目录**、不是仓库根，于是导入旧库：

```
ModuleNotFoundError: No module named 'diffsynth.utils.data.minimax_h3'
```

已在 `VRising-FL2VA.sh` 的 `cd` 之后加：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

> 手敲命令跑 `train.py` 时同样要带。**不要**用 `pip install -e` 覆盖安装来"修"——那会改掉 `/data/danzechen/DiffSynth-Studio` 上的其他工作。

### 5.2 为什么从 HuggingFace 下而不是 ModelScope

初次用 ModelScope 只有 **4-6 MiB/s**，144 GB 要 7 小时。排查后确认瓶颈是网络路径，不是本机：

| 检查项 | 实测 | 结论 |
|---|---|---|
| /nfs 写盘 | 350 MB/s | 不是瓶颈 |
| 下载并发 | 9 连接 / 73 线程 | 不是瓶颈 |
| → 到 `modelscope.cn` RTT | **176 ms** | 跨洋回国内 |
| → 到 `huggingface.co` RTT | **14 ms** | 就近 CDN |

换 HF 后实测 **118 MiB/s**，144 GB 从 7 小时压到 **20 分钟**。

- HF repo id 是 **`MiniMaxAI/MiniMax-H3`**（不是 `MiniMax-AI/`，也不是 ModelScope 的 `MiniMax/MiniMax-H3`）
- 目录结构与 ModelScope 完全一致，四个 pattern 逐个比对过
- 落盘仍到 `<repo>/models/MiniMax/MiniMax-H3/`，保持 DiffSynth 默认 `local_model_path` 布局，训练脚本不用改

### 5.3 两个盘必须分清

| 挂载 | 容量 | 用途 |
|---|---|---|
| `/`（home） | 879G，**仅剩 42G** | ⚠️ `~/.cache/modelscope`、`~/.cache/huggingface` 默认在这，144 GB 会撑爆 |
| `/nfs` | 50T，剩 4.8T | 模型、latent cache 都放这 |

预下载时务必 `export HF_HOME=/nfs/danze/model/.hf_home`。

### 5.4 为什么给 `runner.py` 打了断点续跑补丁

原生 `launch_data_process_task` 没有跳过逻辑，stage 1 跑 32 小时中途挂掉就得从头再来。补丁（`diffusion/runner.py`）：

```python
if os.path.exists(save_path):
    continue
data = model(data)
tmp_path = save_path + ".tmp"
torch.save(data, tmp_path)
os.replace(tmp_path, save_path)   # 同目录 rename 原子，最终文件存在即完整
```

**已验证**：smoke 重跑 64 条从 6.23 s/it 降到 **1.15 s/it**（只剩视频解码），缓存条数/体积不变、无 `.tmp` 残留、抽检 `.pth` 可正常加载。

> 这是对框架代码的主动修改，超出"跑通训练"的字面范围，但 32 小时不可续跑的作业风险不可接受。

### 5.5 为什么加 tensorboard 和 `SAVE_STEPS`

- smoke 那轮完全是**盲跑**：`ModelLogger` 只在开 tensorboard/swanlab 时记 loss，脚本没开 → 5 个 epoch 没有任何 loss 曲线
- 全量单 epoch 要 **52 小时**，只按 epoch 存意味着两天才有第一个可用权重，也无法 early stop

现设 `SAVE_STEPS=2000`（约每 5 小时一存）+ `--enable_tensorboard_log`。

---

## 6. 实测数据

### 6.1 smoke（64 条）全流程

| 阶段 | 耗时 | 产出 |
|---|---|---|
| HF 下载 135 GB | 20 分钟 | 118 MiB/s |
| stage 1 | 6分38秒 | **6.23 s/clip**，cache 746 MB |
| stage 2（5 epoch） | 48 分钟 | **9.06 s/step**，5 × 131 MB |

### 6.2 LoRA 有效性 A/B 验证

`exit 0` 只证明脚本没崩。用 `validate_vrising_lora.py` 做同 seed / 同首尾帧对比：

| | 耗时 | 峰值显存 |
|---|---|---|
| base | 417 s | 63.9 GiB |
| LoRA | 261 s | 64.1 GiB |

**平均像素差 16.3 / 255（6.4%）** —— 明显非零，LoRA 确实生效。产出在 `/nfs/danze/eval/vrising_smoke/`。

### 6.3 全量成本外推（干净卡假设）

| 子集 | stage 1 | cache | stage 2（5 epoch） |
|---|---|---|---|
| smoke (64) | 0.1 h | 0.7 GB | 0.8 h |
| transform (9314) | 16.1 h | 106 GB | 117 h ≈ 4.9 天 |
| **all (20699)** | **32 h** | **236 GB** | **260 h ≈ 10.9 天** |

磁盘够（/nfs 剩 3.6T），**时间是瓶颈**。且当前 GPU 2/3/4/5 都在 100% 跑别的任务，多卡并行短期不现实。

### 6.3b `form_wolf` 实测 —— 外推系统性偏乐观 1.7–1.9×

上表用的是 smoke 那轮的干净卡速率。`form_wolf` 全程与 `zeqingwang` 抢 GPU1，实测：

| | 外推值 | 实测 | 倍数 |
|---|---|---|---|
| stage 1 | 6.0 s/clip | **10.18** | 1.70× |
| stage 1 总耗时 | 2.3 h | **3h55m** | — |
| cache | 15.7 GB | **16 GB** | ✓ 准 |
| stage 2 | 9.06 s/step | **16.67** | 1.84× |
| stage 2 总耗时（6870 步） | 17.3 h | **~32 h**（预计 08-15 下午） | — |

**只有磁盘外推是准的，时间外推全部要乘 1.7–1.9。** 后续排期按实测速率算，
别再用 smoke 那轮的数字——那是当时 GPU1 干净时测的，不代表常态。

### 6.3c 🔴 flow-matching 的 loss 曲线**没有判据性** —— §5.5 的初衷落空了

§5.5 加 tensorboard 的理由之一是「可以看 loss 曲线 early stop」。跑到 5368 步后
读出来的曲线是这样的：

| 步区间 | 0–200 | 1000–1200 | 3000–3200 | 5000–5200 | 5168–5368 |
|---|---|---|---|---|---|
| loss 均值 | 0.1898 | 0.1878 | 0.1834 | 0.1803 | 0.1683 |

**5000 多步只从 0.190 挪到 0.180，而单步噪声远大于这个漂移。** 原因是
flow matching 每步随机采 timestep，loss 的方差主要由 timestep 决定而非模型质量。

**结论：不要用 loss 曲线判断收敛或早停。** §9.4 那条「看 loss 曲线早停」不可行，
选 checkpoint 只能靠采样目视 / 与 GT 的定量比对。tensorboard 仍然值得开——
它能告诉你训练**有没有炸**（NaN、突然抬升），但仅此而已。

### 6.3d Stage A step-5000 的 A/B（08-15，GPU0）

```
base vs step-5000 平均像素差: 15.051 / 255  (5.9%)
base 耗时 366s / LoRA 372s，峰值显存 64.1 GiB
产出: /nfs/danze/eval/vrising_form_wolf_step5000/
```

量级与 smoke 那轮（16.3/255）一致，**LoRA 确实在改变输出**。

但目视比对给出一个**需要留意的负面信号**：抽 0/35/70/106 四帧看，
两版都是可信的 V-Rising 夜间等距场景（土路、木栅栏、植被、狼），
**但 LoRA 版明显更软/更糊**——狼的轮廓和植被细节都不如 base 锐利。

可能是 3.6 个 epoch 在 1374 条上开始过训，也可能只是这一条 clip / 这个 seed 的偶然。
**分辨这两者需要 checkpoint 扫描**（step-500 / 2000 / 5000 同 seed 同 clip 比对），
而这正是 §6.3c 说的——loss 曲线帮不上忙，只能这么做。

> ⚠️ 注意 FL2VA 给了**真实首尾帧**做锚点，两版都被强约束，
> 所以这个对比主要反映的是**中间帧的插值质量**，不是完整的生成能力。

### 6.4 GPU 争抢事件（2026-08-13 09:00 前后）

启动约 2 小时后速率腰斩：

| | 启动时 | 争抢后 |
|---|---|---|
| 速率 | 5.44 s/it | **10.61 s/it** |
| stage 1 ETA | 30h48m | **57h55m** |
| GPU1 显存 | 59.5 GB | 120.8 GB |

原因：另一用户 `zeqingwang`（ReactiveGWM 一作，同项目组）在 GPU1 起了任务占 60.1 GB，与我们的 59.5 GB 抢同一张卡的 SM 时间。

**这是可预见的后果**：占卡脚本原本压着 122 GB，为启动训练把它 SIGTERM 后，我们只用 59.5 GB，剩余约 83 GB 就对外释放了。

当时全卡状态——**没有一张能干净容纳 stage 1（需 ~60 GB）**：

```
GPU0 空闲  8.5GB/  0%    GPU4 空闲 37.1GB/ 82%
GPU1 空闲 24.1GB/100%    GPU5 空闲 36.9GB/ 80%
GPU2 空闲 36.7GB/ 86%    GPU6 空闲 24.9GB/100%
GPU3 空闲 36.9GB/ 87%    GPU7 空闲 52.6GB/  0%   ← 最接近，仍差 7GB
```

> 日志里的 `No minimax_h3_dit models available. This is not an error.` 是**误报**——
> stage 1 本来就不加载 DiT。做异常筛查时不要被它带偏。

**结论**：这是资源协调问题，不是技术问题。与其用技术手段硬扛 20+ 天，不如直接协调错峰。

### 6.5 争抢的持续状态（2026-08-14 09:00 复核）

争抢没有缓解，**已经是常态而非事件**。当前八卡：

```
GPU0  59.9GB/100%   ← zeqingwang            GPU4  99.6GB/ 93%
GPU1 138.9GB/100%   ← 我们 78.8 + 他 60.1   GPU5  99.5GB/100%
GPU2  99.8GB/ 16%                           GPU6  60.0GB/100%
GPU3  99.7GB/100%                           GPU7  59.8GB/100%
```

对方是 `--num_processes 4` 的 ReactiveGWM stage A（GPU 0/1/6/7 各 ~60 GB，`--max-steps 25200`，
`--walltime-seconds 86400`）。**没有一张卡能干净容纳我们的 stage 2（需 ~79 GB）。**

> 顺带修正 §5.3 的余量：`/nfs` 已从 4.8T 降到 **3.6T**（93% used），
> `/`（home）**只剩 39G**（96% used）。home 那 39G 值得清——
> HF / modelscope 缓存默认落在那里，一次误设 `HF_HOME` 就能撑爆。

### 6.6 争抢形态变了（2026-08-15 07:00 复核）—— 显存松了，SM 没松

> ⏱️ **本节描述的是 07:00–08:25 的状态，已被 §6.7 取代**（08:25 争抢完全解除）。
> 保留是因为「显存和 SM 要分开看」这条结论仍然成立，且它是当天能并行起四个作业的依据。

对方从 `--num_processes 4` × ~60 GB 换成了 **`--num_processes 8` × ~25.6 GB**
（ReactiveGWM stage B，`--max-steps 5000`，从 stage_A/checkpoint-25200 起）：

```
GPU0 25.7GB/100%   GPU2 25.6GB/100%   GPU4 25.6GB/100%   GPU6 25.6GB/100%
GPU1 25.6GB/100%   GPU3 25.6GB/100%   GPU5 25.6GB/100%   GPU7 25.4GB/100%
                   （不含我们自己的占用）
```

**这个变化很重要**：八张卡 SM 全部 100%，但**每张卡都还剩 ~118 GB 显存**。
08-14 时「没有一张卡能干净容纳我们的 stage 2（需 ~79 GB）」的判断**不再成立**——
现在**任意一张卡都装得下**。

所以 08-15 能同时起四个作业（GPU0 推理 / GPU4 smoke+action / GPU6 wolf_life stage 1，
外加一个 CPU 探针），这在 08-14 的显存格局下做不到。

**代价仍在 SM 上**：速度还是 16.5x s/it，1.85× 的惩罚一点没变。

### 6.7 🎉 争抢在 08-15 ~08:25 解除 —— 并且暴露出 baseline 本身是脏的

`zeqingwang` 的任务撤出 GPU1 / GPU6，两张卡变成我们独占。速度当场翻倍：

| 作业 | 争抢时 | 独占后 | 倍数 |
|---|---|---|---|
| `form_wolf` stage 2（GPU1） | 16.57 s/it | **7.93 s/it** | **2.09×** |
| `wolf_life` stage 1（GPU6） | 10.46 s/it | **5.21 s/it** | **2.01×** |

从日志能读出准确的切换点（wolf_life 在 elapsed 1:11:53 还是 10.57，1:23:05 已经是 5.20）。

> 🔴 **推论：§6.3 那个「干净卡」基准本身就不干净。**
>
> | | §6.3 记的「干净卡」 | 真正独占 | 差 |
> |---|---|---|---|
> | stage 2 | 9.06 s/step | **7.93** | 慢 14% |
> | stage 1 | 6.0 s/clip | **5.21** | 慢 15% |
>
> 那两个数是 smoke 那轮（08-13）测的，当时 GPU1 上就已经有别人了，只是量小没察觉。
> 所以：
> 1. **真实争抢惩罚是 2.0–2.1×，不是文档记的 1.7–1.9×**
> 2. §6.3 / §7.2 的全部成本外推**都偏悲观约 15%**（对我们有利，不用改决策）
> 3. **教训**：拿「当时手头最快的那次」当基准是不可靠的，除非确认过卡上没有别人。
>    以后记基准速率时应当同时记下 `nvidia-smi` 的全卡快照。

**受益**：两个在跑的作业各提前约 3 小时和 5.5 小时。

| 作业 | 原 ETA | 新 ETA |
|---|---|---|
| `form_wolf` stage 2 | ~13:40 | **~10:51** |
| `wolf_life` stage 1 | ~19:15 | **~13:45** |

> 结论修正：**显存和 SM 要分开看。** 「卡满了」这句话在 08-14 指显存、在 08-15 指 SM，
> 而只有前者会**阻止**作业启动，后者只是让它变慢。排期时别把两者混为一谈——
> 08-14 那种「等一张干净卡」的思路会白白浪费 08-15 这种显存宽松的窗口。

> 磁盘继续下降：`/nfs` **2.7T**（95% used），`/`（home）**38G**（96% used）。
> `all-cache/` 那 83 GB 仍然占着，确认不需要后应当删掉。

---

## 7. 数据规模：`all` + 5 epoch 是过量的（✅ 已按此决策切换）

跑起来之后才算清的一笔账——**当时配置对 LoRA 严重过量**，这比 GPU 争抢更值得重新决策。

> **决策已执行**（08-14 02:41）：终止 `all`，切到 `form_wolf`(1374)。
> 下面保留完整论证，因为换更大子集时同一笔账要重算一遍。

### 7.1 LoRA 容量 vs 训练步数

LoRA 可训参数量（rank 32，`qkv_proj` + `out_proj`，50 层）：

```
qkv_proj  Linear(5376, 21504):  32×5376 + 21504×32 =  860,160
out_proj  Linear(7168,  5376):  32×7168 +  5376×32 =  401,408
每层 1,261,568  ×  50 层  =  63,078,400 ≈ 63M
```

bf16 下 126 MB，与实测 checkpoint 131,227,832 B（125.1 MiB）吻合 ✓

而 `all` × 5 epoch = **103,495 步**（batch size 1）。典型 LoRA 风格适配 1000–10000 步即收敛。
**所以现在不只是慢，是在做边际收益接近零的事。**

### 7.2 各子集成本（干净卡，按实测 6.0 s/clip 与 9.06 s/step）

| 子集 | n | stage 1 | cache | stage 2 (5ep) | 合计 |
|---|---|---|---|---|---|
| **form_wolf** | 1374 | 2.3 h | 15.7 GB | 17.3 h | **< 1 天** |
| form_wolf + ability_wolf_space | 2247 | 3.8 h | 26 GB | 28 h | ~1.3 天 |
| 狼的完整生命周期¹ | 5430 | 9.1 h | 62 GB | 68 h | ~3.2 天 |
| transform | 9314 | 16.1 h | 106 GB | 117 h | ~5.6 天 |
| **all** | 20699 | 32 h | 236 GB | 260 h | **13 天** |

¹ `form_wolf` + `transform_vampire_to_wolf` + `transform_wolf_to_vampire`

完整分类成本见下（n / stage1 / cache / stage2-5ep）：

```
form_vampire              3397   5.7h  38.8GB  42.7h
transform_vampire_to_wolf 2028   3.4h  23.2GB  25.5h
transform_wolf_to_vampire 2028   3.4h  23.2GB  25.5h
transform_rat_to_vampire  1558   2.6h  17.8GB  19.6h
transform_vampire_to_rat  1558   2.6h  17.8GB  19.6h
form_bear                 1516   2.5h  17.3GB  19.1h
form_wolf                 1374   2.3h  15.7GB  17.3h
form_rat                  1238   2.1h  14.1GB  15.6h
ability_vampire_q         1192   2.0h  13.6GB  15.0h
transform_bear_to_vampire 1071   1.8h  12.2GB  13.5h
transform_vampire_to_bear 1071   1.8h  12.2GB  13.5h
ability_wolf_space         873   1.5h  10.0GB  11.0h
form_transfer              647   1.1h   7.4GB   8.1h
ability_bear_q             602   1.0h   6.9GB   7.6h
ability_vampire_space      546   0.9h   6.2GB   6.9h
```

### 7.3 为什么窄切片对 Stage B **更合适**（不只是将就）

- **动作空间干净**：漫游只有 WASD，没有 Q / Space / LeftControl 那些稀疏事件，"动作是否真的在控制画面"一眼可判
- **无形态突变**：单一形态，不受变身时的外观剧变干扰
- **步数落在合理区间**：1374 × 5 = 6870 步，正是 LoRA 的舒适区
- **先例**：ReactiveGWM 本身也是窄域（单个格斗游戏）

**代价（必须说清）**：只学到狼的外观，vampire / bear / rat 不覆盖；变身能力完全没有。
若最终目标包含变身，狼切片的 Stage A 撑不住，需要用"完整生命周期"那一档。

### 7.4 ⚠️ 缓存按**行号**索引，换子集不能复用

缓存文件名 `{data_id}.pth` 里的 `data_id` 就是 metadata 的**行号**（`enumerate(dataloader)`）。
换子集 = 换行序 = **同一个 `{i}.pth` 指向完全不同的 clip**，缓存彻底不可复用。

实测：已缓存的前 1051 条全部是 `ability_bear_q`(602) + `ability_vampire_q`(449)，
而 `form_wolf` 在 `h3_meta_all.jsonl` 里的下标是 10011–11384，**零重叠**。

> 若确实需要跨子集复用，可以写脚本按行号映射搬运 `.pth`（`all` 的第 i 行 ↔ 子集的第 j 行），
> 但前提是所需的行已经缓存过。当前情况下不成立。

### 7.4b ✅ 修正：**前缀保序**的超集可以直接复用缓存（08-15 实证）

上面「换子集 = 缓存彻底不可复用」说得太绝对了。真正的条件是 **`data_id` 映射是否保持**，
而 `data_id` 只是行号，所以只要新 metadata 的**前 N 行与旧 metadata 逐字段相同**，
前 N 条缓存就原样有效。

`h3_meta_wolf_life.jsonl`(5430) 正是这样构造的 —— `form_wolf`(1374) 原序在前，
后面追加 4056 条 transform。两个前提都实测确认过：

```python
wl[:1374] == fw                      # ✅ 逐字段相同（不是只比 video 字段）
runner.py:113  shuffle=False         # ✅ data_process 不打乱，data_id 严格等于行号
```

（注意 `runner.py:67` 的**训练** dataloader 是 `shuffle=True`，只有 data_process 那条是 False。）

做法：把 `form_wolf-cache/0/*.pth` 拷进 `wolf_life-cache/0/`，
§5.4 的断点续跑补丁会把前 1374 条直接跳过。

| | 从零重跑 | 复用缓存 |
|---|---|---|
| 需编码 | 5430 条 | **4056 条** |
| 耗时（按实测 10.18 s/clip） | 15.4 h | **~11.5 h** |

实测：前 1374 条 14 分 28 秒跳完，随后从 `data_id=1374` 开始以 9.98 s/it 编码。
**跨边界核对**（这是复用是否成立的最终判据）：

```
行 1373  form_wolf                 wolf/clip_001374.mp4              ← 复用的，带 action_cond
行 1374  transform_vampire_to_wolf vampire_to_wolf/clip_000001.mp4   ← 新编码的
行 1383  transform_vampire_to_wolf vampire_to_wolf/clip_000010.mp4   ← 与日志正在处理的那条逐字对上
```

> 🔴 **两条约束，破一条就全废**：
> 1. **必须单进程**（`num_processes=1`）。多进程时 `save_path` 是
>    `output_path/{process_index}/{data_id}.pth`，且每个进程 `enumerate` 的是自己那一片，
>    `data_id` 变成片内下标，映射立刻失效。`form_wolf` 那轮是单进程，所以能复用。
> 2. **新增的行只能追加在后面**，不能插在中间或重排前缀。
>
> 推论：**以后建 metadata 时应当刻意保持前缀保序**，把最可能先跑的窄子集放在最前。
> 这是纯粹免费的——只是 `build_h3_metadata.py` 里拼接顺序的问题，
> 却能把「换更大子集」的代价从全量重跑降到只编码增量。§7.4 那条教训因此可以放宽为：
> 子集**范围**可以事后扩大，但**行序**必须在第一次 stage 1 之前定死。

**实际付出的代价**：`all` 终止时已缓存 7263 条（20h、83 GB），全部作废，
`form_wolf` 的 1374 条从零重跑。比 08-13 记录这条时预计的 1654 条大了 4.4 倍。

> 🔴 **教训写在这里**：子集必须在 stage 1 启动**之前**定死。
> 中途改主意的代价 = 已缓存的全部条数，而且随时间线性增长。
> 下次换子集（比如上"狼的完整生命周期"）前，先把 §7.1 那笔步数账算完再开跑。
> `all-cache/` 那 83 GB 现在是纯占位，确认不再需要后可以删掉回收 /nfs 空间。

### 7.5 建议（已采纳）

**切到 `form_wolf`，把 Stage A 当作 Stage B 的验证床。**

当前真正的未知数不是"画风学得够不够好"，而是**动作注入通路对不对**——
分段池化的时序对齐、bias 有没有避开 keyframe anchor、action 能否活着进 cache。
这些用 1374 条一天就能验证，用 20699 条要等两周才知道答案。

通路验证通过后，再决定 Stage A 值不值得投更大的数据。

> 事后看这个判断是对的，而且**回报比预期早**：切换当天（08-14）就完成了
> 动作数据管线 + 架构改造 + 缓存注入三件事（全是 CPU 工作，与 stage 1/2 并行，见 §9.4）。
> 如果还在跑 `all`，这些工作同样能做，但要等到 08-26 才有 GPU 验证它们。

---

## 8. 怎么跑

```bash
cd /nfs/danze/repo/DiffSynth-Studio-new

# metadata（已存在可跳过）
python3 /nfs/danze/data/v_rising/build_h3_metadata.py

# 冒烟
STAGE=1 bash examples/minimax_h3/model_training/lora/VRising-FL2VA.sh smoke

# 正式（driver 带自动重试 + 阶段串联；GPU/SUBSET/SAVE_STEPS 可配）
GPU=1 SUBSET=form_wolf setsid nohup bash /nfs/danze/run_h3_all.sh \
  > /nfs/danze/logs/h3_form_wolf_nohup.log 2>&1 &
kill $(cat /nfs/danze/logs/h3_form_wolf.pid)     # 中断

# 动作注入（stage 1 完成后跑，纯 I/O）
python3 /nfs/danze/inject_action_into_cache.py --subset form_wolf --dry-run
python3 /nfs/danze/inject_action_into_cache.py --subset form_wolf

# 注入落点探针（纯 CPU，秒级，改了序列布局或注入代码后必跑）
python3 /nfs/danze/probe_action_injection.py --cache <...>-cache --n 9999

# 带动作的 stage 2
CUDA_VISIBLE_DEVICES=4 STAGE=2 ACTION_BUTTONS=12 ACTION_TRAIN_ONLY=1 SAVE_STEPS=50 \
  bash examples/minimax_h3/model_training/lora/VRising-FL2VA.sh smoke

# 复用缓存跑更大的前缀保序子集（§7.4b）——必须单进程
cp -n <小子集>-cache/0/*.pth <大子集>-cache/0/
CUDA_VISIBLE_DEVICES=6 STAGE=1 bash .../VRising-FL2VA.sh wolf_life
```

> 🔴 **`validate_vrising_lora.py` 必须从仓库根目录跑。** `ModelConfig` 的
> `local_model_path` 是相对 cwd 解析的（`./models/MiniMax/MiniMax-H3/...`），
> 在别处跑会找不到本地那 135 GB 权重，转去 ModelScope 重下（实测 408 kB/s），
> 而且它持有的 `~/.cache/modelscope/hub/.lock/MiniMax___MiniMax-H3` 会**把其他作业一起堵住**。
> 症状是别的作业日志里刷 `Still waiting to acquire lock on ...`。

日志/pid 文件名都带 `$SUBSET`，所以换子集不会覆盖上一轮的日志。

`VRising-FL2VA.sh` 支持 `STAGE=1|2|all` 和 `SAVE_STEPS=N`。拆开阶段的意义：transformer 66 GB 和编码器 78 GB 可以分别调度，且 stage 2 能带 cache 存在性检查独立重跑。

### 日志

| 文件 | 内容 |
|---|---|
| `/nfs/danze/logs/h3_form_wolf.log` | driver 主日志，阶段切换与重试 |
| `h3_form_wolf_stage1.log` / `_stage2.log` | 训练输出 |
| `h3_form_wolf.pid` | 中断用 |
| `h3_all*.log` | 已终止的 `all` 那轮，保留作速率参照 |

> 进度条是 `\r` 刷新的，`tail` 直接看是一坨。用 `tr '\r' '\n' < 日志 | tail`。

---

## 9. 未决问题

### 9.1 ✅ 已关闭

| 问题 | 结论 |
|---|---|
| 是否切到 `form_wolf` | 已切（08-14 02:41），代价 83 GB / 20h 见 §7.4 |
| cache 能否事后注入 action | 能，已实测跑通并校验，见世界模型文档 §5 |
| Stage B 的时机 | 全是 CPU 工作，已与 Stage A 并行完成 |
| **动作注入通路对不对** | **08-15 已验**：落点、开销、梯度、通道语义全部通过（世界模型文档 §7.0）。剩时序对齐 |
| **换更大子集要不要全量重跑 stage 1** | **不用**，前缀保序时可复用，见 §7.4b |
| **能不能用 loss 曲线早停** | **不能**，flow matching 的 loss 没有判据性，见 §6.3c |

### 9.2 ✅ 已执行：没有等 baseline（08-15）

08-14 的安排是等 stage 2 跑完 32 小时拿到 baseline 曲线，再开 `--action_num_buttons` 对比。
08-15 插队在 smoke 上验掉了，`form_wolf` 那轮 baseline 照常跑，两边不互相阻塞。

⚠️ 判据的**准确说法**：`--action_train_only` 会冻结包括 LoRA 在内的一切，
所以它对比的是**底模 baseline**，不是"纯 LoRA baseline"。别把这两条曲线搞混。

> 🔴 **但回头看，「比 loss 曲线」本来就不是好判据**，理由有二：
> 1. flow matching 的 loss 方差压倒性地大（§6.3c），两条曲线重不重合根本读不出来
> 2. 训练 dataloader 是 `shuffle=True` 且没有 `set_seed`（`runner.py:67`），
>    两次运行的数据顺序不同，逐步比对本来就不成立
>
> **真正给出结论的是两个不依赖 loss 的证据**：
> `probe_action_injection.py` 验空间落点，存盘权重验梯度与逐通道语义。
> 都在世界模型文档 §7.0a / §7.0c。以后设计这类「通路对不对」的验证，
> 优先找**确定性的、可断言的**判据，而不是看曲线像不像。

### 9.3 🔴 GPU 争抢

见 §6.4 / §6.5。属资源协调问题，与 `zeqingwang` 错峰比任何技术手段都有效——
当前 1.85× 的损失，没有任何代码优化能追平。

### 9.4 待定

1. **Stage A 是否跑满 5 epoch。** ~~可看 loss 曲线早停~~ —— **这条不可行**（§6.3c）。
   step-5000 的目视结果还偏负面（比 base 更糊，§6.3d）。
   **改为：做 checkpoint 扫描**（step-500 / 2000 / 5000 同 seed 同 clip），
   最好再加一个与 GT 中间帧的定量比对，否则没有客观依据选权重。
2. ~~**`form_wolf` 只覆盖狼**，升级到 5430 要全量重跑~~ —— **已启动，且不需要全量重跑**。
   `wolf_life`(5430) stage 1 于 08-15 07:12 启动，复用了 `form_wolf` 的 1374 条缓存，
   只编码新增 4056 条（§7.4b），ETA ~18:40。
   仍然只覆盖狼，vampire / bear / rat 的外观没有；但**变身**这一环补上了。
3. **12 键里有 7 键在 `form_wolf` 上近乎恒零**（见世界模型文档 §3.5）。
   `wolf_life` 会好一些——`transform_*` 带 LeftControl 和 remap 出的 Alpha1，
   但 Alpha2/Alpha3（bear / rat）在这个子集上仍然恒零。
4. **🔴 时序对齐仍未验**（世界模型文档 §7.0d）。这是 Stage B 剩下的唯一风险点，
   也是全部现有证据都碰不到的一条，需要带动作的端到端生成。
   `wolf_life` stage 1 完成后应当优先做这件事。
5. **`all-cache/` 的 83 GB** 确认不需要后应当删掉，`/nfs` 只剩 2.7T（95%）。

---

## 10. 文件索引

| 路径 | 说明 |
|---|---|
| `.../transform_classified/train.jsonl` | 原始训练集（20699 条） |
| `.../transform_classified/h3_meta_{smoke,form_wolf,transform,all}.jsonl` | H3 metadata |
| `/nfs/danze/data/v_rising/build_h3_metadata.py` | metadata 转换 |
| `.../lora/VRising-FL2VA.sh` | 训练入口（STAGE / SAVE_STEPS 可配） |
| `/nfs/danze/run_h3_all.sh` | driver（重试 + 串联，GPU/SUBSET 可配） |
| `/nfs/danze/h3_action.py` | 动作 sidecar → latent 时间轴张量（带 self-test） |
| `/nfs/danze/inject_action_into_cache.py` | 把 action 事后注入 stage 1 缓存 |
| `/nfs/danze/probe_action_injection.py` | 注入落点探针（纯 CPU，世界模型文档 §7.0a） |
| `/nfs/danze/validate_vrising_lora.py` | LoRA A/B 验证（`--subset` / `--lora` / `--outdir` 可配） |
| `.../smoke_lora_baseline_backup/` | 原纯 LoRA 冒烟权重的备份，见世界模型文档 §6.2 的警告 |
| `diffsynth/diffusion/runner.py` | 已打断点续跑补丁 |
| `/nfs/danze/finetune_minimax_h3.py` | ⚠️ **占卡脚本，与微调无关**（`-g GPU -m GiB`，用 SIGTERM 停） |

> 仓库里 4 个文件有未提交改动（`runner.py`、`minimax_h3_dit.py`、
> `minimax_h3_audio_video.py`、`train.py`）+ 1 个未跟踪（`VRising-FL2VA.sh`）。
> 都是本项目的改造，不要 `git checkout` 掉。

---

## 11. 变更记录

| 日期 | 变更 |
|---|---|
| 08-13 | 初版：数据选型、两阶段训练、环境运维、`all` 方案成本核算 |
| 08-14 | **决策执行**：终止 `all`(7263/20699)，切 `form_wolf`。**新增** §6.3b（实测比外推慢 1.7–1.9×）、§6.5（争抢现状 + 磁盘余量修正）、§9.2（不必等 baseline）。**更新** §0 进度表、§7.4（真实沉没成本）、§8 跑法与日志 |
| 08-15 晚些 | **新增** §6.7（争抢 ~08:25 解除，速度翻倍 2.0–2.1×；**并修正 §6.3 的「干净卡」基准本身是脏的**，真实惩罚 2.0–2.1× 而非 1.85×，全部成本外推偏悲观 ~15%）、§6.6 标注为已被取代。**更新** §0 进度表与 ETA。**新增**第三份姊妹文档 `minimax_h3_architecture_and_data.md`（架构 / 数据契约 / 帧率重采样），并在 §3 加指针（107 帧只覆盖 4.46s，尾部丢 9 帧，32.7% 重复帧） |
| 08-15 | **Stage B 验证完成 + `wolf_life` 启动。** **重要修正** §7.4b：前缀保序的超集**可以复用缓存**，`wolf_life`(5430) 因此只需编码 4056 条而非 5430 条（15.4h→11.5h），并推论出「以后建 metadata 应刻意保持前缀保序」。**新增** §6.3c（flow-matching loss 曲线无判据性，§5.5 早停初衷落空）、§6.3d（step-5000 A/B：像素差 15.05/255 但目视更糊）、§6.6（争抢形态变化：显存松了 SM 没松，显存/SM 要分开看）。**关闭** §9.2（已在 smoke 上验掉，且记下「比 loss 曲线」本就不是好判据）。**更新** §0 进度表、§9.1/9.4、§8 跑法（新增探针、带动作 stage 2、缓存复用；补 `validate_vrising_lora.py` 必须从仓库根跑的坑）、§10 文件索引 |
