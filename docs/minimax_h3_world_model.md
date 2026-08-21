# 把 MiniMax-H3 改造成动作条件游戏世界模型

> 更新：2026-08-15
> 骨干：`diffsynth/models/minimax_h3_dit.py`　参考实现：`ReactiveGWM_Code`（Wan2.2-TI2V-5B）
> 姊妹文档：[`minimax_h3_vrising_finetune.md`](minimax_h3_vrising_finetune.md)（数据、常规微调、运维决策）
> 　　　　　[`minimax_h3_architecture_and_data.md`](minimax_h3_architecture_and_data.md)（架构、数据契约、帧率重采样）
> 　　　　　[`minimax_h3_abot_data.md`](minimax_h3_abot_data.md)（ABot-500h 新数据源的处理方案）
>
> 📌 §7.0d「时序对齐仍未验」是针对 **V-Rising** 说的（那边动作时间轴是复刻 H3
> 内部 `round()` 映射推断出来的）。ABot 那条线上视频与动作用**同一个显式帧号列表**
> 截取，对齐是构造性的且已实测复核（argmin 扫描 8/8 通过），见
> [`minimax_h3_abot_data.md`](minimax_h3_abot_data.md) §3.2 / §3.4。

本文回答：**为什么这样改，而不是别的改法。**

---

## 0. 结论

ReactiveGWM 的「玩家动作 → per-block additive bias」可以移植到 H3，
但 H3 骨干与 Wan2.2 差异很大，有 **三处必须重写**，外加 **一个两阶段缓存的硬约束**。

| 阶段 | 内容 | 风险 | 状态（08-15） |
|---|---|---|---|
| **A** | FL2VA LoRA，学画风 + 形态先验 | 无 | 🔄 stage 2 进行中（step 5368/6870） |
| **B** | 动作 additive bias，chunk 级控制 | 中，**主要收益在这** | ✅ **已跑通并验证**，只剩时序对齐待验 |
| **C** | `denoise_mask_video` 做 history-conditioned rollout | 中高，「世界模型」的分界线 | 未开始 |
| **D** | causal 化 + KV cache，实时交互 | 高 | 未开始 |

**Stage B 已落地的部分**（§4 是实录，不再是清单）：三处必须重写全部完成，
两阶段缓存约束已满足，`form_wolf` 的 1374 条缓存里 `action_cond` 实测存在且非零。

**08-15 更新：运行时行为已验证。** §7.0 原先列的三条待验里，两条已经关闭
（注入落点见 §7.0a 的探针，开销见 §7.0b 的实测），第三条（时序对齐）仍未验。
Stage B 现在可以称为「跑通」，但**还不能称为「对」**——见 §7.0c。

---

## 1. H3 骨干：和 Wan 差在哪

```
video latents ─┐
audio latents ─┼─→ 单条 packed sequence [1, S, 5376] ─→ 50 × DiTBlock(self-attn + MLP) ─→ final_layer
text embeds  ─┘         ↑ token_tags 区分模态              ↑ AdaLN 按模态分 3 组调制
```

`num_layers=50`, `hidden_size=5376`, `num_heads=56`, `head_dim=128`, `patch_size=(1,2,2)`。

序列布局（`_build_packed_fl2va`）：

```
[ text | cond(keyframe anchor) | audio | video | pad ]
         └────── img_pos 同时覆盖这两段，cond 在前 ──────┘
```

### 差异 1：单流 self-attn，**没有 cross-attention**

50 个 block 里只有 `attn`(self) + `mlp` + `adaln_proj`，不存在 Wan 那样的 `cross_attn(x, context)`——文本是**拼进同一条序列**参与 self-attn 的。

**推论**：

- 玩家动作走 additive bias —— **照搬可行**，H3 这边甚至更干净（没有 gate 位置需要斟酌）
- NPC 策略走 cross-attn —— **没有现成模块可挂**。两条路：
  1. 新建 cross-attn 层（改动大，且与 H3 的单流设计相悖）
  2. **把策略编码成 token 拼进 packed sequence**（推荐）——符合 H3 设计哲学，白嫖已有 self-attn

> 若走方案 2，策略 token 应复用 `token_tags=1`（text 模态）。
> **不要**新开 tag=3 并把 `MINIMAX_H3_ADALN_MODALITY_NUM` 从 3 改成 4——
> 那样 `adaln_out_features = 6×5376×3 = 96768` 就对不上预训练权重了。

### 差异 2：token 散落在 flat 序列

ReactiveGWM 的注入是整块相加：

```python
bias = emb.unsqueeze(2).expand(-1, -1, h * w, -1).reshape(B, f * h * w, dim)
return x + bias
```

H3 里 hidden 是 `[S, 5376]`，video 行位置由 `img_pos` 给出，而且 **`img_pos` 前 `cond_rows_count` 个是 keyframe anchor，不是视频帧**（见 `model_fn_minimax_h3` 的 `x[0].index_copy_(0, img_pos[cond_rows_count:], video_rows)`）。

整块加会把 bias 糊到 keyframe anchor 上、污染首尾帧条件。当初的设计写法是：

```python
video_pos = img_pos[cond_rows_count:]            # [latent_t * frame_rows]
bias = emb.repeat_interleave(frame_rows, dim=0)  # [latent_t, C] -> [latent_t*frame_rows, C]
hidden = hidden.index_add(0, video_pos, bias)
```

**实现时换成了更省的写法。** 序列布局 `[text | cond | audio | video | pad]` 里 video 段是
**连续**的，所以不需要 `index_add` 的散射语义，直接 `narrow` 出来广播相加即可
（省掉 `repeat_interleave` 物化的 134 MB/层，见 §8）：

```python
v0 = int(img_pos[cond_rows_count])               # video 段起点，已跳过 anchor
video = hidden.narrow(0, v0, latent_t * fr).view(latent_t, fr, -1) + emb.unsqueeze(1)
hidden = torch.cat([hidden.narrow(0, 0, v0),
                    video.reshape(-1, C),
                    hidden.narrow(0, v0 + latent_t * fr, rest)], dim=0)
```

> ⚠️ 这里依赖「video 段连续」这条假设。它对 FL2VA 布局成立（已由 `_build_packed_fl2va` 确认），
> 但如果以后改了序列布局，这段会**静默算错**而不是报错。改布局时记得回来看这里。

### 差异 3：时间轴**非均匀** —— 最容易踩

```python
video_latent_t = ((num_frames - 5) // 17) * 5 + 2
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)     # _T_GROUP = 5
```

每组 5 个 latent token 覆盖 17 个原始帧：**第一个覆盖 1 帧，后 4 个各覆盖 4 帧。**

`num_frames=107` 验证：`latent_t = (102//17)*5+2 = 32`，而 `6×17 + (1+4) = 107` ✓
且已由真实 cache 交叉验证：`input_latents.shape = (1,24,**32**,30,52)`。

**所以 ReactiveGWM 的 `F.adaptive_max_pool1d(keyboard, output_size=f)` 均匀池化在 H3 上会系统性错位**——每组第一个 token 本该 1:1 对应单帧，被抹成约 3.34 帧的窗口。动作是稀疏按键脉冲，这种错位直接损坏控制信号的时序对齐。

```python
def frame_spans(latent_t):
    spans = [_FRAME_PER_TOKEN[k % 5] for k in range(latent_t)]
    ends = list(itertools.accumulate(spans))
    return list(zip([0] + ends[:-1], ends))

def bin_to_latent(mat, latent_t):            # [num_frames, K] -> [latent_t, K]
    return torch.stack([mat[s:e].amax(0) for s, e in frame_spans(latent_t)])
```

用 `amax` 而非 mean：按键是二值脉冲，窗口内按下过就算按下。取均值会把单帧脉冲
稀释成 0.25 这种没有物理含义的强度。

已实现于 `/nfs/danze/h3_action.py`，带 self-test，六个 `num_frames` 全部通过：

```
num_frames= 107  latent_t= 32  spans 累加= 107 ✓
num_frames=  90  latent_t= 27  spans 累加=  90 ✓   （73/56/39/22 同样 ✓）

107 帧前 10 个 token 区间: (0,1) (1,5) (5,9) (9,13) (13,17) (17,18) (18,22) ...
                            ↑ 每组第一个只覆盖 1 帧，这正是均匀池化抹掉的东西
```

> 依据说明：`_FRAME_PER_TOKEN` 是 RoPE 的时间跨度约定，我由它 + latent_t 公式反推出帧映射，
> 未直接读 video VAE 源码。但 cumsum 精确复现 `num_frames`、且 `latent_t` 与实测缓存吻合，是强证据。

### 差异 3b：帧率重采样 —— 实现时才发现，文档原先漏了

**这是本文档 08-13 版本的一个真实遗漏。** 上面的分箱只解决了「107 帧 → 32 token」，
但 sidecar 根本不是 107 帧：

```
sidecar:  81 帧 @16fps            （原始录制）
video:   107 帧 @24fps            （fix_frame_rate=True 重采样后取的）
```

直接拿 81 帧的动作矩阵去按 107 帧的 spans 分箱，**动作时间轴会被压缩 1.32 倍**——
第 81 帧的按键会落到第 81 个视频帧的位置上，而它实际对应第 107 帧。
这是整段系统性错位，比均匀池化的错位更严重。

修正办法是在分箱**之前**先把动作重采样到视频时间轴，且映射必须与视频帧的取法**逐帧一致**
（复刻 `core/data/operators.py` 的 `FrameSamplerByRateMixin.map_single_frame_id`）：

```python
j       = torch.arange(num_frames, dtype=torch.float64)
raw_idx = torch.round(j / target_fps * raw_fps).long().clamp(max=T_raw - 1)
mat     = mat[raw_idx]                       # [81, 12] -> [107, 12]
```

`raw_fps` 从 sidecar 的 `timestamp` 反推而不是写死 16.0，这样换录制帧率不用改代码。

> 完整链路：`sidecar → frames_to_matrix → resample_to_video_timeline → bin_to_latent`。
> 三步缺一不可，`action_from_sidecar()` 把它们串成一步。

---

## 2. 硬约束：两阶段缓存白名单（✅ 已满足）

`split_pipeline_units`（`diffusion/training_module.py:316`）在 stage 1 末尾挂了 `GeneralUnit_RemoveCache`：

```python
required_params = list(loss_required_params) \
                + [i for i in inspect.signature(self.pipe.model_fn).parameters] \
                + [各 unit 的 fetch_input_params()]
```

**不在白名单的 key 会被剪掉**，stage 2 读不到，动作张量白算。

**最省事的解法：给 `model_fn_minimax_h3` 加 `action_cond=None` 形参。** 加了就自动进白名单，不用碰 `GeneralUnit_RemoveCache`。

这是整个改造里最不直观、也最省事的一步，**务必先做**——否则 stage 1 跑 32 小时到 stage 2 才发现动作丢了。

> ✅ 已实现（`minimax_h3_audio_video.py:812`）。
> ⚠️ **必须是具名形参，`**kwargs` 不算数**——白名单来自
> `inspect.signature(...).parameters`，而 `**kwargs` 在签名里只有一个 `kwargs` 条目，
> `action_cond` 这个 key 仍会被剪掉。这一点容易想当然写错。

---

## 3. 动作数据（已实测核对）

### 3.1 两种 sidecar 形态

`transform_classified/<cat>/clip_XXXXXX.json` 有两种结构，逐帧字段一致：

| 形态 | 出现在 | 结构 |
|---|---|---|
| `list[81]` | 静态形态（`vampire`、`bear`…） | 直接是帧数组 |
| `{metadata, frames[81]}` | 技能、变身类 | 帧在 `frames` 键下 |

每帧：`{frame_index, timestamp, active_keys, click_events}`。

> ⚠️ 解析时必须同时处理两种形态。早期抽样脚本只认 `list` 而静默跳过 `dict`，
> 曾误判"Space / LeftControl 缺失"，实际数据是完整的。

### 3.2 各类别的按键（每类抽样 60 条）

| 类别 | 观察到的按键 |
|---|---|
| `vampire_space` / `wolf_space` | WASD + **Space** |
| `vampire_q` / `bear_q` | WASD + **Q** |
| `transform_*` | WASD + **LeftControl** |
| `form_vampire` | WASD + Mouse0 + LeftShift + Space + R |

全集出现过：`W A S D Mouse0 Space LeftControl LeftShift Q R F`。

### 3.3 为什么 `forward_ctrl_remap` 是必需的

`processed_data/dataset_info.json` 定义了 12 键 schema 和一个 remap：

```
button_cols: W A S D Mouse0 Space LeftControl LeftShift Q Alpha1 Alpha2 Alpha3
forward_ctrl_remap: vampire_to_wolf→Alpha1,  vampire_to_bear→Alpha2,  vampire_to_rat→Alpha3
```

**Alpha1/2/3 在原始录制里根本不出现**——变身用的是 **LeftControl**，而 LeftControl 本身**不区分变成哪个形态**。remap 按目标形态把它消歧成 Alpha1/2/3。

没有这一步，模型学不到「按哪个键变成哪个形态」的因果，只能学到「按 LeftControl 会变成某种东西」。**所以这个 remap 是语义必需，不是可选装饰。**

### 3.4 数据从哪来

`processed_data/actions/*.parquet` **不在本机**（`dataset_info.json` 的 `source` 指向 `/opt/dlami/nvme/...`）。
两条路：从那台机器同步，或**按同一 schema 从 `clip_XXXXXX.json` 重新生成**。
后者更可靠——sidecar 就在本地且完整。

> ✅ 走了后者，实现在 `/nfs/danze/h3_action.py`。`BUTTON_COLS` 与
> `dataset_info.json` 的 12 键 schema 逐位对齐，顺序不能改（改了就和已注入的缓存对不上）。
> schema 之外的键（F / E / R / Mouse1 / Tab）直接丢弃；`click_events` 里的鼠标事件
> 并入 `Mouse0`。

### 3.5 ⚠️ `form_wolf` 的实际按键分布：12 键里只有 5 键活着

注入前用全部 1374 条算了一遍覆盖率（出现过该键的 clip 数）：

| 键 | clip 数 | 占比 |
|---|---:|---:|
| W | 899 | 65.4% |
| A | 866 | 63.0% |
| D | 849 | 61.8% |
| S | 826 | 60.1% |
| **Space** | **459** | **33.4%** |
| LeftShift | 8 | 0.6% |
| Mouse0 | 6 | 0.4% |
| Q | 3 | 0.2% |
| LeftControl | 2 | 0.1% |
| Alpha1 / Alpha2 / Alpha3 | **0** | **0.0%** |

1374 条里 1372 条有按键，2 条全零（静止片段，正常）。

**两点结论**：

1. **[姊妹文档](minimax_h3_vrising_finetune.md) §7.3 的「动作空间干净」被证实了**，
   而且比预期还干净——WASD + Space 五个通道，其余七个近乎恒零。控制信号是否生效会非常好判。
2. **Alpha1/2/3 精确为 0**，符合预期：`form_wolf` 里没有变身，`forward_ctrl_remap`
   在这个子集上不会触发。**所以这一轮验证不到 remap 的正确性**（§3.3），
   那要等上 `transform_*` 子集才能验。

保留 12 键而不是缩到 5 键，是为了 schema 跨子集稳定——缓存里的 `action_cond` 宽度
一旦定死，换子集时不想再动 `enable_action_conditioning(num_buttons)` 和已注入的数据。
代价是这一轮有 7 个通道的 embedder 收不到梯度信号（输入恒零 ⇒ 权重不动），
属于可接受的浪费（每层 7×5376 个参数）。

---

## 4. 改动实录（已实现）

四处改动，其中**两处偏离了原设计**，偏离的理由写在下面。

### 4.1 `diffsynth/models/minimax_h3_dit.py`

**偏离 1：`action_embedders` 不能建在 `__init__` 里。**

原设计直接在 `__init__` 写 `self.action_embedders = ...`，实现时发现会炸：
`core/loader/model.py:89` 用 `load_state_dict(assign=True)` 加载预训练 DiT，
**`strict` 默认为 True**，凭空多出的 `action_embedders.*` 会直接 `Missing key(s)` 报错。

改成延迟建立（`minimax_h3_dit.py:303`）：

```python
def enable_action_conditioning(self, num_buttons: int):
    ref = next(self.parameters())            # 借现成参数拿 device/dtype
    self.num_action_buttons = num_buttons
    self.action_embedders = nn.ModuleList(
        [nn.Linear(num_buttons, self.hidden_size, bias=False) for _ in range(self.num_layers)]
    )
    for m in self.action_embedders:
        nn.init.zeros_(m.weight)             # 零初始化：起点严格等价于原模型
    self.action_embedders.to(device=ref.device, dtype=ref.dtype)
    return self
```

`__init__` 里只留 `self.num_action_buttons = 0`，它同时充当注入开关——
forward 里 `inject_action = action_cond is not None and self.num_action_buttons > 0`，
所以**缓存里带着 `action_cond` 但没开 `--action_num_buttons` 时会整个跳过**，
不会误注入。当前 Stage A 那轮正是这个状态。

**偏离 2：用 `narrow` + 广播 + `cat`，不用 `index_add`。** 理由见 §1 差异 2。

```python
for i, block in enumerate(self.blocks):
    if inject_action:
        emb = self.action_embedders[i](action_cond.to(hidden.dtype))   # [latent_t, C]
        video = hidden.narrow(0, v0, n_video).view(latent_t, fr, -1) + emb.unsqueeze(1)
        hidden = torch.cat([hidden.narrow(0, 0, v0),
                            video.reshape(n_video, -1),
                            hidden.narrow(0, v0 + n_video, hidden.shape[0] - v0 - n_video)], dim=0)
    hidden = gradient_checkpoint_forward(block, ...)
```

进循环前做一次越界检查（`v0 + n_video > hidden.shape[0]` 直接抛），
避免形状对不上时静默算错。

> `_repeated_blocks = ["MiniMaxH3DiTBlock"]` 被 offload / fp8 机制用来识别可重复块。
> `action_embedders` 不在 block 内部，不受影响。

### 4.2 `diffsynth/pipelines/minimax_h3_audio_video.py`

- `model_fn_minimax_h3(..., action_cond=None)` —— 具名形参，**过缓存白名单的关键**（§2）
- 透传 `action_video_start=int(img_pos[cond_rows_count])` 和 `action_frame_rows=(h//2)*(w//2)`

原设计说要改 `_build_packed_fl2va` 的返回值补 `cond_rows_count` / `frame_rows`，
**实际不用改**：`cond_rows_count` 在 `model_fn` 里本来就算过（`cond_anchor.shape[0]`），
`frame_rows` 由 latent 尺寸直接得出。少动一个函数，少一处出错面。

### 4.3 `examples/minimax_h3/model_training/train.py`

原设计走「metadata 加 `action` 字段 + `special_operator_map` 加 loader」，
**实际改走了缓存注入路线**（§5），所以 `parse_extra_inputs` 一行没动，metadata 也不用改。
train.py 只加了两个开关和一段初始化：

```python
--action_num_buttons N     # 0 表示关闭；动作张量本身来自缓存的 action_cond
--action_train_only        # 冻结除 action_embedders 外的一切（选项 3）
```

初始化位置有**两个**约束，都不能挪：

- 在 `from_pretrained` **之后**——否则 strict 加载报 Missing key（见 4.1）
- 在 `switch_pipe_to_training_mode` **之后**——它内部先 `freeze_except` 冻结一切，
  之后新建的参数默认 `requires_grad=True`，正好是我们要训的

并且 `if ... and getattr(self.pipe, "dit", None) is not None` 跳过 stage 1
（`sft:data_process` 不加载 DiT）。

### 4.4 metadata：不用改

缓存注入路线下 metadata 保持原样，`--data_file_keys` 也不用加 `action`。
sidecar 路径由 `video` 字段同名推导（`wolf/clip_000001.mp4` → `wolf/clip_000001.json`）。

---

## 5. cache 复用：不必重跑 32 小时（✅ 已跑通）

Stage 1 产出的 cache 不含 action。但**可以事后注入**——`.pth` 存的是普通结构：

```
tuple(3):
  [0] inputs_shared  dict: imgvid_cond_noise_aug, audio_cond_noise_aug,
                           use_gradient_checkpointing(_offload),
                           input_latents (1,24,32,30,52), audio_input_latents (2,32,178),
                           keyframe_cond_anchor (780,96)
  [1] inputs_posi    dict: prompt_embeds (906,5120), packed{img_pos, audio_pos, text_pos,
                           img_position_ids, token_tags, cu_seqlens, seq_len}
  [2] inputs_nega    dict: {}
```

遍历 `.pth`，往 `inputs_shared` 加一个 `action_cond` key 再存回 —— **纯磁盘 I/O**，
比重跑 stage 1 便宜两个数量级。注入脚本沿用 `.tmp` + `os.replace` 的原子写法，
避免中断留下半截文件。

### 实测结果（`form_wolf`，`/nfs/danze/inject_action_into_cache.py`）

```
$ python3 -c "import torch; d=torch.load('.../form_wolf-cache/0/0.pth', weights_only=False); ..."
inputs_shared keys: [imgvid_cond_noise_aug, audio_cond_noise_aug,
                     use_gradient_checkpointing, use_gradient_checkpointing_offload,
                     input_latents, audio_input_latents, keyframe_cond_anchor,
                     action_cond]                      ← 注进去了
action_cond: (32, 12)  float32  sum=34.0               ← 形状对、非零
```

脚本内置三道校验，任何一条不过就 `SystemExit` 而不是继续写坏数据：

1. `data_id >= len(rows)` → 缓存和 metadata 行序对不上
2. `act.shape != (latent_t, 12)` → 分箱出错
3. **`shared["input_latents"].shape[2] != latent_t`** → 拿真实 latent 交叉验证动作时间轴，
   这条最有价值：它把 §1 差异 3/3b 那一整套推导和真实 VAE 输出对上了

⚠️ **缓存目录是两层的**（`form_wolf-cache/0/0.pth`，不是 `form_wolf-cache/0.pth`），
脚本用 `os.walk` 递归找 `.pth`。手写路径时容易踩。

### 前提（两条都必须满足）

1. `model_fn_minimax_h3` 已有**具名形参** `action_cond`（§2）
2. 传进来的 metadata 必须和跑 stage 1 时是**同一个文件、行序不变**——
   `{data_id}.pth` 的 `data_id` 就是行号

> 第 2 条与姊妹文档 §7.4 是同一件事的两面：**同一子集内**可事后注入 action；
> **跨子集**则因行号错位不可复用。

---

## 6. 训练策略

`action_embedders` 是**新增参数，LoRA 训不到**。三个选项：

1. `--trainable_models` 带上 action_embedders + `--lora_base_model dit`（推荐，等价于 ReactiveGWM 的 scoped module finetuning）
2. 全量 finetune DiT（~32B 参数，需 zero3，显存吃不消）
3. 只训 action_embedders、冻结其余（收敛慢但最安全）

**建议先用零初始化 + 选项 3 验证注入通路**，然后再放开到选项 1。
选项 3 已实现为 `--action_train_only`。

### 6.1 ⚠️ 判据的准确说法（原文这里写得不严谨）

原文说"若 loss 曲线与**纯 LoRA baseline** 一致"。但 `--action_train_only` 的实现是
`for p in self.parameters(): p.requires_grad_(False)`，**LoRA 也一起冻了**。
所以它对比的是**底模 baseline**，不是纯 LoRA baseline。

两者都能验通路——零初始化保证 bias 恒为 0、前向逐位等价——但别把两条曲线搞混：

| 跑法 | 可训参数 | 对比基准 |
|---|---|---|
| `--action_train_only` | 只有 action_embedders（3.2M） | **底模** |
| 选项 1（LoRA + action） | LoRA 63M + action_embedders | 纯 LoRA baseline |

### 6.2 ✅ 不必等 32 小时的 baseline（08-15 已执行）

零初始化的判据**不需要完整曲线**：权重全零 ⇒ 前向输出逐位等价于不注入 ⇒
只要头几十步 loss 与基准重合，通路就是通的。这个信号 smoke（64 条）一小时就能拿到。

08-14 的排期是等 `form_wolf` 的 32 小时 stage 2 跑完再验，
**是拿 32 小时换一个 1 小时能拿到的信号**。08-15 插队执行：

```bash
python3 /nfs/danze/inject_action_into_cache.py --subset smoke
CUDA_VISIBLE_DEVICES=4 STAGE=2 ACTION_BUTTONS=12 ACTION_TRAIN_ONLY=1 SAVE_STEPS=50 \
  bash examples/minimax_h3/model_training/lora/VRising-FL2VA.sh smoke
```

结果见 §7.0。**回头看，「跑一轮看 loss 曲线」还不是最好的判据**——真正给出结论的是
§7.0a 的探针（落点）和 §7.0c 的存盘权重（梯度与通道语义），两者都不依赖 loss 曲线。
loss 曲线在 flow matching 下方差极大、几乎读不出信息（姊妹文档 §6.6）。

> ⚠️ 顺带一个坑：smoke 的 `output_path` 与之前那轮纯 LoRA 冒烟**是同一个目录**，
> `on_epoch_end` 会用只含 action_embedders 的权重覆盖掉原来的 `epoch-*.safetensors`。
> 已备份到 `model/minimax_h3_vrising/smoke_lora_baseline_backup/`。
> 换实验配置时记得先看 `output_path` 里有没有别的东西。

`action_embedders` 参数量：`12 × 5376 × 50 = 3,225,600`（3.2M），
只有 LoRA 63M 的 5%，显存和速度上都不构成负担。

---

## 7. 缺口（如实说明）

### 7.0 Stage B 的运行时验证（08-15，三条待验关掉两条）

08-14 时这三件事**只有推理，没有运行时证据**。08-15 在 smoke（64 条）上跑了一轮
`--action_num_buttons 12 --action_train_only`，外加一个 CPU 探针，结果如下：

| 待验 | 状态 | 依据 |
|---|---|---|
| bias 确实避开了 keyframe anchor | ✅ **关闭** | §7.0a 探针，1374+64 条全通过 |
| 前向不 OOM / 不显著变慢 | ✅ **关闭** | §7.0b，16.54 vs 16.57 s/it |
| 分箱 + 重采样的时序对齐 | 🔴 **仍未验** | §7.0c，需要带动作的端到端生成 |

#### 7.0a 注入落点：用探针验，不要指望训练

「bias 有没有加错位置」**训练是验不出来的**——加错了 loss 照样下降，
只是学到的东西没有物理意义。这是个会静默失败的问题，必须单独验。

`/nfs/danze/probe_action_injection.py`（纯 CPU，秒级）用**真实缓存的 packed 元数据**
复刻 `model_fn` 的 `v0` 推导，然后验六件事：

1. `narrow`+广播+`cat` 的结果与原设计 `index_add` 参考实现**逐位相同**
   —— 这才是「video 段连续」那条假设（§1 差异 2 标记为会静默算错）的真正证明
2. 被改动的行集合**恰好等于** `img_pos[cond_rows_count:]`，一行不多一行不少
3. 780 行 keyframe anchor 逐行未被触碰
4. text / audio / pad 区未被写入
5. 第 k 个 latent token 的 390 行拿到的确实是 `emb[k]`，无错位
6. 零初始化下输出与不注入逐位相同

```
form_wolf-cache: ✅ 1374/1374 通过
smoke-cache    : ✅   64/64 通过
S=14464~14528  v0=1984~2046  latent_t=32  fr=390  n_video=12480  anchor=780
```

> `v0` 在 1984–2046 之间随 prompt 长度浮动，说明推导是动态的、没有写死 —— 这一点
> 本身也值得记：如果 `v0` 是常数，反倒说明取错了东西。

#### 7.0b 开销：可以忽略

| | form_wolf baseline（GPU1） | smoke + action（GPU4） |
|---|---|---|
| 速度 | 16.57 s/it | **16.54 s/it** |
| 显存 | 78.8 GB | **77.7 GB** |

3.2M 参数、bias 不物化（§8），实测与分析一致。**注入不构成负担，不用为它做取舍。**

#### 7.0c 梯度确实流经注入点 —— 且逐通道语义完整

`--action_train_only` 下 `export_trainable_state_dict` 按 `requires_grad` 过滤，
所以存盘的 `step-50.safetensors` **只含 action_embedders**：

```
6,456,064 B ≈ 3,225,600 参数 × 2 (bf16)     50 个张量，层索引 0..49
离开零初始化的层: 50/50                      各层 |w|max 中位 2.9e-03
```

50 层全部离开零初始化 ⇒ 梯度在**每一个 block** 都流过了注入点。若 bias 加到了
pad 或不影响 loss 的位置，梯度会是 0。

**更强的信号在逐通道上**。`nn.Linear(bias=False)` 的 `dW = dYᵀ·x`，输入恒零的通道
拿不到梯度，所以梯度非零的通道集合**必须精确等于数据里出现过的按键集合**：

| 通道 | smoke 里出现的 clip 数 | 梯度 \|w\|max |
|---|---:|---|
| W / A / S / D | 47 / 39 / 37 / 37 | 3.1e-03 / 3.0e-03 / 2.9e-03 / 3.1e-03 |
| LeftControl | 32 | 2.7e-03 |
| Space / Q / Mouse0 | 12 / 4 / 3 | 1.6e-03 / 1.1e-03 / 1.2e-03 |
| **Alpha1 / Alpha2 / Alpha3** | **9 / 3 / 7** | 1.7e-03 / 7.8e-04 / 1.5e-03 |
| **LeftShift** | **0** | **0.000e+00** ← 恒零输入，精确为零 |

11 个出现过的键梯度全部非零，唯一没出现的 `LeftShift` 精确为 0。**逐项对上。**
这证明动作张量的**逐列语义**从 sidecar → 分箱 → 缓存 → 模型全程没有错位或串列。

> 🎁 意外收获：**`forward_ctrl_remap` 第一次得到了端到端验证。** smoke 是从全量
> 20699 条里随机抽的，覆盖了 `transform_*`，所以 Alpha1/2/3 非零（9/3/7 条）。
> §3.5 说「这一轮验证不到 remap」，那是针对 `form_wolf` 说的；smoke 恰好补上了。
> 注意这只证明 remap 的**输出形态**正确到达了模型，不证明模型学会了那个因果。

**这一轮已跑完整 5 epoch（320 步）**，权重轨迹是干净的单调增长：

| checkpoint | step-50 | 100 | 150 | 200 | 250 | 300 | 320 |
|---|---|---|---|---|---|---|---|
| `\|w\|max` | 3.14e-03 | 6.74e-03 | 9.64e-03 | 1.27e-02 | 1.65e-02 | 1.98e-02 | **2.16e-02** |
| 非零层 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | **50/50** |

**梯度在整个训练过程中持续流动**，不是只在头几步动一下然后死掉（那会是注入被
某种方式截断的典型症状）。增长近似线性、无震荡、无饱和，说明 3.2M 参数在
64 条数据上远未训满——符合预期，这一轮的目的是验通路而不是学动作。

产出：`/nfs/danze/model/minimax_h3_vrising/smoke/step-*.safetensors`（7 个，各 6.46 MB）。

#### 7.0d 仍然没验的：时序对齐

**剩下这一条是风险最高的**（§1 差异 3b 就是漏了才发现的），而且上面全部证据都碰不到它：
探针验的是空间落点，梯度验的是通道语义，**都不验「第 k 个 latent token 对应的是不是
第 k 段时间」**。唯一的验法是带动作的端到端生成，看按键与画面动作是否同步。

**所以 Stage B 现在是「跑通且落点正确」，不是「完成」。**

### 7.1 长程 rollout

H3 是**双向全序列 DiT**，一次生成 107 帧就结束，而世界模型需要无限滚动。

好消息：`model_fn_minimax_h3` 已有 `input_latents_video` + `denoise_mask_video`（Retake 机制在用）——**把前 k 个 latent token 设为 clean 历史（mask=0），后面重新去噪（mask=1）**，就能一段段往前滚。这块比 Wan 那边好做。

坏消息：每滚一步都要跑完整 50 层双向 attention，没有实时性。

### 7.2 真正的实时 AR

需要 causal attention + KV cache。当前是 `_sdpa_varlen_attention` 双向全连接，且 RoPE 时间轴非均匀（`_FRAME_RESCALE = 5/3`），causal 化后位置编码要重新推导。改动量最大，放最后。

### 7.3 音频分支的税 —— **不值得动**

世界模型不需要音频，但 audio token 占着序列和算力。然而：

```
audio_rows = 178 × 2 = 356      vs      video_rows = 32 × 390 = 12480
```

**不到 3%**。而拆掉它要连带改 `_build_packed_fl2va` 布局、`token_tags`、AdaLN 索引、`audio_pos`，伤筋动骨。

> 结论：音频保持静音兜底，别动。

---

## 8. 序列长度参考（107 帧 / 480×832）

| 项 | 值 |
|---|---|
| `latent_t` | 32 |
| `latent_h, latent_w` | 30, 52 |
| `frame_rows` | 390 |
| video rows | 12480 |
| audio rows | 356 |
| cond rows（首尾帧） | 780 |
| text rows | 变长（实测某条 906） |
| `seq_len` | 上述之和向上对齐到 64 |

action bias：`[32,12] → embedder → [32,5376] → 广播到 [32,390,5376]`，每层一次、共 50 层。

⚠️ 原设计的 `repeat_interleave(390)` 物化后是 `12480×5376×2 bytes ≈ 134 MB/层`（bf16）。
**实现里用 `emb.unsqueeze(1)` 广播规避了**，bias 本身不物化。

> 但要说清规避到什么程度：**相加的结果张量仍然要物化**（`narrow`+广播加产生
> 12480×5376，`cat` 再产生完整 seq ≈ 147 MB）。省掉的是 bias 那一份，不是全部。
> 这在 autograd 下无法避免——`index_add_` 的原地形式会破坏反向图。
> 好在有 gradient checkpointing，这些中间量在反向时重算而不是全程驻留。

---

## 9. 文件索引

| 路径 | 说明 |
|---|---|
| `diffsynth/models/minimax_h3_dit.py:303` | `enable_action_conditioning`，延迟建 embedder |
| `diffsynth/models/minimax_h3_dit.py:405` | forward 里的注入点 |
| `.../minimax_h3_audio_video.py:539` | `_build_packed_fl2va` 序列布局 |
| `.../minimax_h3_audio_video.py:812` | `model_fn_minimax_h3`，`action_cond` 具名形参 |
| `examples/minimax_h3/model_training/train.py:71` | 动作层初始化（位置有两个约束，§4.3） |
| `/nfs/danze/h3_action.py` | sidecar → `[latent_t, 12]`，带 self-test |
| `/nfs/danze/inject_action_into_cache.py` | 缓存事后注入（三道校验） |
| `/nfs/danze/probe_action_injection.py` | 注入落点探针（纯 CPU，六道校验，§7.0a） |
| `diffsynth/diffusion/training_module.py:316` | `split_pipeline_units`，缓存白名单 |
| `ReactiveGWM_Code/inference/models/dit.py:173` | `WanModelAction`，移植参考 |
| `ReactiveGWM_Code/training/bidirectional/` | 双向训练 / 缓存预计算参考 |
| `.../transform_classified/<cat>/clip_*.json` | 动作 sidecar（两种形态） |
| `processed_data/dataset_info.json` | 12 键 schema + `forward_ctrl_remap` |

---

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 08-13 | 初版：设计论证，三处必须重写 + 两阶段缓存约束 |
| 08-14 | Stage B 实现完成。**新增** §1 差异 3b（帧率重采样，原设计遗漏）、§3.5（`form_wolf` 按键分布）、§5 实测、§7.0（运行时未验）。**修正** §1 差异 2（改用 narrow+广播）、§4（清单→实录，记两处偏离）、§6.1（判据表述不严谨）、§8（物化规避的真实程度） |
| 08-15 晚些 | §7.0c **补完**：smoke+action 已跑满 5 epoch / 320 步，权重单调增长 3.14e-03 → 2.16e-02，50/50 层全程非零 ⇒ 梯度持续流动而非只动头几步 |
| 08-15 | **Stage B 跑通并验证。** §7.0 从「全部未验」改写为实录：新增 §7.0a（注入落点探针，1374+64 条全通过，证明「video 段连续」假设成立）、§7.0b（开销实测 16.54 vs 16.57 s/it）、§7.0c（50/50 层梯度非零；逐通道梯度与数据按键覆盖精确对应，`LeftShift` 恒零输入 ⇒ 梯度精确为 0；`forward_ctrl_remap` 借 smoke 首次端到端验证）、§7.0d（时序对齐仍未验，是剩余最高风险）。**新增** `probe_action_injection.py` 到 §9 |
