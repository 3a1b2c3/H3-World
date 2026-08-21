# 逐 latent 动作文本注入 · 管线状态

> 最后更新 2026-08-21 · 方案见 `docs/action_text_injection_plan.html` · 标注核对台见 `docs/action_prompt_viz.html`

## 决定

- **放弃 `action_cond` 特征空间注入**（加性 bias 与 FiLM 均已实测失效），只走逐 latent 动作文本注入。
- FiLM 训练已于全局约 5028/9840 步停止，12 个 checkpoint 保留在 `output/minimax_h3_abot/7872_film/` 作对照。
- 条件表示为 **9 位按键的纯函数**：`W A S D I J K L F`，训练与推理走完全同一条路径。

---

## 一、方案要点

标注模板固定，只换两个槽，两个槽都是 9 位按键的确定性函数：

```
头部（原样保留）  <Picture 1>: [首帧视觉 392 行] + 场景描述 80–260 行
逐 latent（37 条） the man <怎么动>, camera <相机视角怎么动>
```

头部不另设主体锚点：场景描述本来就描述了角色，`the man` 有指代物；且沿用它意味着
训练与推理是同一次 `presentation_fl2va` 调用，不需要额外对齐。

| 槽 | 输入位 | 规则 |
|---|---|---|
| 怎么动 | W S A D | 按固定次序拼接；全 0 → `stands still`；**W∧S、A∧D 相消**（`bin_to_latent` 对按键取 amax，4 帧窗口里先按 W 后按 S 会让两位同时为 1） |
| 相机 | J / L | `pans left` / `pans right`，J∧L 相消 |
| | F | 速度档：F=1 → `sharply`，F=0 → `slowly` |
| | I / K | `tilts down` / `tilts up`（实测 K 的 `d_pitch` 95% 为正、I 95% 为负），I∧K 相消，可与摇镜叠加 |
| | 无相机键 | 有移动键 → `follows him`；无移动键 → `holds steady` |

### 第 9 位 F 的来源与必要性

原始录像只有 8 个键。**方向能从 IJKL 读出（命中率 0.85–0.97），速度读不出来**：

| 问题 | 实测（600 条 clip） | 结论 |
|---|---|---|
| 按 J 的步里 slowly / sharply | 0.66 / 0.34 | 接近抛硬币 |
| 速度随按住时长变化？ | 第 0 步 0.122 → 第 1–5 步 0.203 → 第 6 步+ 0.176 | 第 0 步偏低是分箱边界效应（键在 bin 中途按下），之后即平台 |
| 速度是片段级常量？ | ICC ≈ 0.475，整段档位一致仅 38.9% | 都不是 |

F 由 **COLMAP 实测摇镜速率**合成：按了 J 或 L 且逐帧 `|d_yaw| ≥ 0.225` 时置 1。众数查表预测器准确率 **0.700 → 0.871**（去掉速度档只用 4 键是 0.890，但相机从句会退化成 IJKL 的确定性函数，COLMAP 那 6 个通道就白测了）。

被舍弃的一类：`drifts <方向>`（角色没动而相机自己在平移，占 1.7%），按定义无法从按键推出，并入 `holds steady`。这是训练与推理之间已知且不可消除的小失配。

### 词表

400 条 clip × 37 步实测：**60 种标注 = 9 种移动从句 × 16 种相机从句的实际共现**，单条 clip 内平均 5.0 种。整个数据集只需约 60 次文本编码，一次算完缓存即可。

---

## 二、版面与绑定

序列布局（`latent_t=37`、`latent 30×52`、`frame_rows=390`）：

```
[ 图像 pad 392 | 场景描述 80–260 | 37 条标注 ~421 | cond 390 | audio 414 | video 14430 | pad ]
  └──── 头部 472–652（重新编码）────┘
                    └──────────── text_len 796–1157（均值 971）────────────┘
seq_len 15744–15872 → 16064–16448（均值 16237）    注意力开销 +5.4%
```

**镜像偏移**：标注 k 落在 `t = (text_len − 视频跨度 − 1) + s_k`，与第 k 帧的 t 偏移**恒为 −201**，且全部 `t < text_len`。

> 为什么不直接放到帧的 t 上：视频的 t 轴原点就是 `text_len`（`_video_grid(latent_t, frame, float(text_len))`），放上去必然越过它。而预训练里文本与视频的 t 偏移**恒为负**，越过去会出现零和正值——那是模型没见过的配置，且落在低频带里（高频档在几百的跨度上早已混叠）。镜像偏移把绑定信号换成"往回看固定距离"一条统一规则，不变量不破。

**硬绑定掩码**（承重墙）：自注意力对 token 置换等变，光把 37 条标注塞进序列，模型无从知道哪条对哪帧；位置只提供软偏置。`score_mod` 让**标注行 k 与第 k 帧以外的 video 行互相不可见**，其余一律不动——video×video、锚点×全部 video、标注×标注、标注×cond 全部保持原样，因此去掉标注行即精确回到原模型。

跨帧上下文不受影响：文本行在 scatter 进主序列**之前**先过 2 层 `token_refiner` 的全连接自注意力（`refiner_cu = [0, text_len, text_len]`），37 条标注彼此已充分混合。

---

## 三、代码改动（已完成并验证）

| 文件 | 改动 |
|---|---|
| `code/abot/action_script.py` | 规则表：`keys9()` 导出 9 位；`annotate_from_keys9()` 纯函数 |
| `code/abot/inject_abot_text.py` | **新建**。事后重写 `prompt_embeds` + `packed`，删除 `action_cond` |
| `diffsynth/models/minimax_h3_dit.py` | `_build_action_block_masks()`；`_sdpa_varlen_attention` 透传 `block_masks`；forward 建一次给 50 层共用 |
| `diffsynth/pipelines/minimax_h3_audio_video.py` | `_build_packed_fl2va(..., action_text_spans=)` 镜像偏移落位 + 导出 `action_text_rows/video_start/frame_rows`；`model_fn` 透传 |
| `code/scripts/ABot-FL2VA.sh` | `ACTION_MODE=text\|cond\|none`（默认 text）；text 模式下 `--action_num_buttons` 恒为 0 |
| `code/abot/probe_action_mask.py` | **新建**。掩码落点探针 |
| `code/abot/check_page_js.py` | **新建**。可视化页的运行时检查（假 DOM 跑 script） |
| `code/abot/verify_text_cache.py` | **新建**。缓存完整性校验，六条判据，支持分片 |
| `diffsynth/pipelines/...` 推理侧 | `PromptEmbedder` 接 `action_script` 并输出 `action_text_spans`；`PackedSequenceBuilder` 接 spans；`__call__` 新增 `action_script` / `negative_action_script` |
| `code/abot/infer_abot.py` | 识别纯 LoRA checkpoint 为 `mode=text`；从 9 键生成标注；新增 `--cfg-scale`；负 prompt 走动作零参考 |

原文件备份在会话 scratchpad。

### 验证结果

**掩码落点探针**（`probe_action_mask.py`，4/4 通过）

```
[1] 不传 action_text_rows 与原路径逐位相同   ✓
[2] 标注 k ↔ 帧 k 放行 / ↔ 帧 j≠k 屏蔽       ✓
[3] 其余全部未被触碰                         ✓
[4] 与显式布尔掩码参考实现一致 max|Δ|=5e-7   ✓
```

**版面**：不传 spans 时不含新键；图像+锚点段与文本段以外逐位未改；`token_tags`/`img_pos` 未变；与自己那帧的 t 偏移唯一值 `−201`。四类构造错误全部拦截（越界、重叠/乱序、条数不符、块太靠后）。

**数据构建**（7872 条全量，8 卡分片约 10 分钟）

头部（首帧视觉 + 场景描述）每条**重新编码**，走 `presentation_fl2va(prompt, image_counts)`
—— 与推理侧同一次调用，训练/推理由构造保证一致，也不依赖缓存里原有的行。实测 0.25 s/条。

全量逐条校验（`verify_text_cache.py`）**7872/7872 通过，0 失败**：

```
头部（图像 392 + 场景描述）  472–652 行（均值 556）
text_len                    796–1157（均值 971）
seq_len                     15744–15872 -> 16064–16448（均值 16237，+5.4%）
镜像偏移取值集合             [-201.0]      ← 7872 条全同
action_cond                  已全部删除
```

> 踩过的坑：早期版本沿用缓存里已有的图像 pad 行，做推理侧对接时才发现两边头部来源不同；
> 而且已改写的 2400 条把场景描述行覆盖掉了，缓存里恢复不出来。改成每次从源数据重新编码后，
> 被损坏的条目自动修复，幂等性也更强（结果只取决于源数据）。

**推理侧**（`smoke_text` / `smoke_text_cfg5`，基座 + 从 FiLM checkpoint 剥出的纯 LoRA）

```
cfg=1.0   50/50 步  2.97 s/it   生成 8.3 MB
cfg=5.0   50/50 步  5.32 s/it   生成 9.3 MB（正负两侧，负 = 动作零参考）
推理侧版面几何: 镜像偏移 -201，与训练侧全量实测一致
```

**掩码开销**（16k 真实序列）

- `create_block_mask` 构建：首次 `0.175 s`（编译），之后 **`12.3 ms`**。每个 forward 建一次给 50 层共用，相对训练步是 0.12%，可忽略。BlockMask 稀疏度 1.5%。
- **注意力本身变慢 23%**：`2.41 s/it` → `2.97 s/it`。一旦传 `block_mask`，整条注意力就从 flash 切到 FlexAttention（`attention_forward` 的 `compatibility_mode` 分支），哪怕只屏蔽 1.5% 的块。训练侧按此推算 `9.65 s/it` → 约 `11.9 s/it`，10 epochs 从 26 小时变约 32 小时。这是硬绑定的实际代价，暂时接受。

---

## 四、怎么跑

```bash
# 数据构建（视频 latent 不动，只重写文本条件；换标注方案重跑此步即可）
python3 code/abot/inject_abot_text.py \
    --meta data/abot_meta_train_7872.jsonl \
    --cache output/minimax_h3_abot/7872-cache --dry-run     # 先试算
python3 code/abot/inject_abot_text.py --meta ... --cache ... # 再写入

# 训练（stage 1 若已有缓存可跳过；ACTION_MODE 默认 text）
NUM_PROCESSES=8 STAGE=2 OUT=output/minimax_h3_abot/7872_text \
    bash code/scripts/ABot-FL2VA.sh 7872

# 掩码探针（改动 DiT 后必跑）
python3 code/abot/probe_action_mask.py
```

---

## 五、未完成 / 待办

1. **判据工具未写。** `compare_action_ckpt.py` 在零参数方案上失效（没有动作层可差分）。新判据：**只改第 k 条标注，测输出的哪些帧变了**——变化集中在第 k 帧及邻域即绑定生效。注意 37×37 注意力对角占优在硬掩码下是架构保证的，恒为真，不能当判据。
2. **显存未实测。** `seq_len` 均值 16237（原 15808），且 FlexAttention 的显存特性与 flash 不同，stage 2 起训前要跑一次确认。停训前是每卡 86 GB / 143.7 GB，余量应当充足。
3. **`64-cache` 是两关键帧配置**（`keyframe_cond_anchor [780,96]`、`action_cond [37,17]`），与 7872 的单关键帧不一致，做验证时不要拿它当基准。
4. **训练时的动作脚本 dropout 未实现。** CFG 的负 prompt 是显式写出的动作零参考，不依赖 dropout 即可工作（探针实测弱方向被放大约 9 倍），但 dropout 能让模型对零动作的响应更规范，值得后续加。

---

## 六、零训练探针的结论（方案第 0 步）

基座模型，同首帧同 seed，只改 prompt 尾句。判据用相位相关估全局水平位移，符号约定已标定（`cum_dx` 与 `Σd_yaw` 相关系数 **r = −0.936**，14/14 反号，`cum_dx` 为负 = 相机左摇）。

| 样本 | left − right | 像素 MAE | 判定 |
|---|---:|---:|---|
| `a3ad9c24` | −116 | 43.26 | ✓ 强 |
| `33b38f0f` | −4 | 13.56 | ✗ 空 |
| `2c75323e` | −116 | 50.22 | ✓ 强 |

**2/3 强阳性。** 且动作零参考负 prompt 把弱方向放大约 9 倍（`right` 相对基线 +14 → **+131**），正是对付无响应样本的机制。注意用**空**负 prompt 的 CFG 反而更差（效应量 2.53× → 1.96×），因为它放大的是整段 prompt 的服从度而非动作差量。

方案第 0 步通过。
