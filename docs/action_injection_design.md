# 动作注入方案分析 —— 针对 MiniMax-H3 架构

> 2026-08-20　基于 `output/minimax_h3_abot/7872/step-{500,1000}.safetensors` 的实测诊断，
> 参考 LingbotWorld-2（本地源码）、Matrix-Game 2.0（论文）、Incantation（GitHub）三个方案。

本文回答一件事：**当前的加性 bias 注入为什么失效，以及针对 H3 这套架构，什么注入方式最合理。**

结论先行：

- 当前方案不是"没学到"，是**注入量级只有 hidden 的 0.5‰**，且**梯度在互相抵消**。
- 根因不是学习率，是**加性 bias 与画面内容无关**这一表达能力缺陷。
- 针对 H3 的架构，最合理的是**在 `norm1` 之后、与 AdaLN 并列的位置做 per-latent-token 的 FiLM，配零初始化 gate**。
- 三个参考方案里，能直接用的只有 LingbotWorld 的 FiLM；Matrix-Game 的 cross-attn 和
  Incantation 的文本接口都卡在 H3 的架构约束上。

---

## 1. 当前方案失效的实证诊断

现状（`minimax_h3_dit.py:405-424`）：每层一个 `Linear(8, 5376)`，零初始化，
输出作为加性 bias 加到 video 段每个 latent token 上。

```python
emb = self.action_embedders[i](action_cond)          # [latent_t, 5376]
video = hidden.narrow(0, v0, n_video).view(latent_t, fr, -1) + emb.unsqueeze(1)
```

### 1.1 注入量级：hidden 的 0.5‰

用真实 stage-1 缓存的 latent 过 `video_patch_proj`，得到注入点处的 hidden：

```
hidden 每行 L2      224.55        每元素 |x| 中位 1.0131（RMSNorm 量级，正常）
step-500  单层 bias L2  0.0947    相对 hidden 0.042%
step-1000 单层 bias L2  0.1521    相对 hidden 0.068%
```

50 层累积也救不回来 —— 层间 bias 方向的余弦相似度只有 **+0.17 ~ +0.20**（基本不相关），
累积走的是随机游走 `√50 ≈ 7` 倍而非同向叠加的 50 倍：

```
0.1521 × √50 = 1.076  →  相对 hidden 仅 0.479%
```

**动作条件对残差流的扰动是千分之五。** 画面变化几乎全部来自 LoRA 和首帧条件。

### 1.2 按当前配置训完也不会有效果

`--num_epochs 3 × 984 步 = 2952 步`。按实测增速（500→1000 步，列 L2 ×1.606）外推到训完，
bias L2 约 0.7–0.8，**相对 hidden 仍只有 0.35%**。要达到 5% 的有效调制量（L2≈11.2），
需要现在的 **74 倍**。

### 1.3 关键：梯度在互相抵消（这条决定了不能靠调学习率解决）

对比 step-500 与 step-1000 的权重差分：

| 指标 | 实测 |
|---|---|
| 每步实际更新（每元素） | `1.92e-06` |
| Adam 理论步长（lr=1e-4） | `1.00e-04` |
| **实际 / 理论** | **1.92%** |
| ΔW 与已有 W 的方向余弦 | **+0.064**（几乎正交） |
| `\|W₅₀₀\| + \|ΔW\|`（若同向） | 3.927 |
| 实际 `\|W₁₀₀₀\|` | **2.866**（明显抵消） |

每步只走出理论步长的 1.92%，说明**梯度在不同 batch 之间方向不一致**，Adam 的动量互相抵消。

**根因**：加性 bias 是 `Δx = b(action)`，b 只依赖按了哪个键，**与当前画面内容无关**。
但"按 D 键"的视觉效果显然依赖当前朝向和场景 —— 面朝东按 D 和面朝北按 D，画面变化完全不同。
不同样本要求 b 指向不同方向，平均下来就抵消了。

> 这是**表达能力**问题，不是优化问题。提高学习率只会让一个方向错误的量长得更快。

---

## 2. H3 架构的三个硬约束

任何方案能不能用，由这三条决定。

### 2.1 没有 cross-attention

`MiniMaxH3DiTBlock`（`minimax_h3_dit.py:214-234`）只有 `attn`(self) + `mlp`：

```python
h = self.norm1(x)
h = _modulate_scale_shift(h, shift_msa, scale_msa, combined_indices)
h = self.attn(h, rope_freqs=..., cu_seqlens=...)
x = _modulate_gate(residual, gate_msa, h, combined_indices)
# 同样的结构再走一遍 norm2 / mlp / gate_mlp
```

要走 cross-attn 路线，等于给 50 层各新增一个 cross-attention 模块，在 LoRA 微调场景下
从零训它，是比当前更难的问题。

### 2.2 AdaLN 是粗粒度的，装不下动作信号

```python
combined_indices = inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM(3) + token_tags.clamp(min=0)
unique_timesteps, inverse_indices = torch.unique(timesteps, ...)   # pipeline:863
```

`timesteps` 是 per-token 标量，但取值只有 `t_video` / `t_audio` / `1.0` / `cond_noise_aug`
这么几个，所以 `combined_indices` 只有 **O(10) 组**。
**整个 video 段（14430 个 token、37 个 latent 步）共享同一组 scale/shift/gate。**

这一条否掉了看似最优雅的路线 ——「把 action 加到 `t_emb` 上、复用预训练 `adaln_proj`」：

实测动作的时间变化率（2000 条抽 300）：

```
相邻 latent token 之间动作发生变化的比例   9.5%
整段动作完全不变的 clip                  34 / 300
```

动作在 37 个 latent 步里频繁变化，粗粒度调制会把时序整个抹掉。
**动作必须走独立的 per-latent-token 通道。**

### 2.3 RMSNorm + AdaLN-Zero，调制接口在 norm 之后

`_norm` 是 `nn.RMSNorm`（`minimax_h3_dit.py:43`）。调制顺序是
**norm → scale/shift → 子层 → gate**。

而当前注入在 **block 入口，也就是 `norm1` 之前**。这是位置错配：
模型有一套成熟的调制接口（norm 后的 scale/shift、输出端的 gate），
动作却从接口外面硬塞进 residual 流。RMSNorm 随后会把绝对量级归一化掉，
注入的 bias 量级语义变得模糊。

---

## 3. 三个参考方案的机制

### 3.1 LingbotWorld-2 —— FiLM 调制

源码 `lingbot-world-v2/wan/modules/model_causal.py:369-376`，每个 block 的 self-attn 之后：

```python
c2ws_hidden = cam_injector_layer2(silu(cam_injector_layer1(plucker_emb)))
c2ws_hidden = c2ws_hidden + plucker_emb              # residual
cam_scale = cam_scale_layer(c2ws_hidden)
cam_shift = cam_shift_layer(c2ws_hidden)
x = (1.0 + cam_scale) * x + cam_shift                # ← FiLM
```

- 控制信号是 **6×64 维 Plücker 相机嵌入**，patchify 成 per-token（空间非均匀）
- `cam_scale/shift_layer` 用 **xavier 初始化，非零**
- 每层都注入

### 3.2 Matrix-Game 2.0 —— 双通路

- **离散键盘** → 专门的 **cross-attention** 模块
- **连续鼠标** → concat 到 latent + MLP + temporal self-attention
- **帧级信号**（不做空间变化，与我们相同）
- 优化后**只在前一半 DiT blocks** 注入

### 3.3 Incantation —— 文本即控制接口

- 动作转成自然语言（"Move forward"、"Staff slam"），T5 编码
- 模型内部**没有任何动作注入模块**，复用 Wan2.2 已有的 cross-attention 文本通道
- per-frame 控制是在**推理循环**层面实现的（`inference.py`）：

```python
for fi in range(num_gen_frames):              # 逐 latent 帧自回归
    context = prompt_embeds_list[pi]          # 每帧换一个 prompt（0.25 秒一个动作）
    for si, tsv in enumerate(denoising_steps):
        flow_pred = model(noisy, context=context, kv_cache=kv_cache,
                          current_start=pos * fsl)
```

**causal / streaming 生成 + KV cache**，每帧一个新 prompt。

---

## 4. 适配性评估

| 方案 | 机制 | H3 能否用 | 卡在哪 |
|---|---|---|---|
| **LingbotWorld FiLM** | `(1+scale)*x + shift` | ✅ **能，最合适** | — |
| ↳ per-token 空间非均匀 | Plücker 几何 | ❌ 不适用 | 那是连续相机位姿，几何上本就空间非均匀；我们是 8 维离散按键，全局量，整帧共享才对 |
| ↳ xavier 非零初始化 | — | ❌ 不能照搬 | 它是大规模训练；我们是 LoRA 微调预训练模型，非零初始化一上来就破坏原模型。必须零初始化 |
| **Matrix-Game 键盘 cross-attn** | cross-attention | ❌ | §2.1 H3 无 cross-attn，要给 50 层各加一个新模块 |
| ↳ 只在前一半 blocks | — | ✅ 可借鉴 | 但用**可学习 gate** 让模型自己决定更好，也更符合 H3 的 AdaLN-Zero 风格 |
| ↳ 鼠标 concat 成 token | 序列拼接 | △ 理论可行 | H3 本就是 packed 序列 `[text\|anchor\|audio\|video\|pad]`，但要改 packing、RoPE、`token_tags`、stage-1 缓存格式。改动最大，留作后手 |
| **Incantation 文本接口** | 每帧换 prompt + KV cache | ❌ | H3 是**双向全序列一次性去噪**（37 个 latent 步同时denoise），没有自回归滚动的机会，也没有 KV cache 机制。要搬等于把 H3 改成 causal/streaming —— 架构级重写，不是微调 |
| ↳ 动作编码进 prompt 文本 | — | △ 可作快速 baseline | 零架构改动，但文本整段共享，37 步的动作只能压成一段描述，时序精度差 |

---

## 5. 推荐方案

**在 `norm1` 之后、与 AdaLN 并列的位置，做 per-latent-token 的 FiLM，配零初始化 gate。**

```python
# MiniMaxH3DiTBlock.forward 内
h = self.norm1(x)
h = _modulate_scale_shift(h, shift_msa, scale_msa, combined_indices)   # 原有 AdaLN

if a_scale is not None:                        # 新增：per-latent-token 动作调制
    hv = h.narrow(0, v0, n_video).view(latent_t, fr, -1)
    hv = hv * (1 + a_scale.unsqueeze(1)) + a_shift.unsqueeze(1)
    h = torch.cat([h.narrow(0, 0, v0),
                   hv.reshape(n_video, -1),
                   h.narrow(0, v0 + n_video, h.shape[0] - v0 - n_video)], dim=0)

h = self.attn(h, ...)
x = _modulate_gate(residual, gate_msa, h, combined_indices)
```

动作侧的模块（对齐 LingbotWorld 的两层 MLP + residual）：

```python
a = self.action_mlp2(F.silu(self.action_mlp1(action_cond))) + self.action_proj(action_cond)
a_scale = self.action_scale[i](a) * self.action_gate[i]     # 零初始化
a_shift = self.action_shift[i](a) * self.action_gate[i]
```

### 五条理由，逐一对应架构事实

1. **位置对齐预训练接口**（§2.3）—— 模型本来就在这个点接受 scale/shift 调制，
   权重已适应这种扰动形式；入口处硬加 bias 是它没见过的形式。
2. **RMSNorm 之后量级确定**（rms≈1），`scale=0.05` 确切意味着 5% 调制。
   而 norm 之前注入，bias 会先被 `rms(x)` 除掉，量级语义模糊。
3. **乘性 = 内容相关**，直接解决 §1.3 的梯度抵消：
   `Δh = scale(a) ⊙ h`，同一个 scale 作用在不同 h 上产生不同效果，
   不再要求"一个与画面无关的固定向量"同时满足所有样本。
4. **零初始化仍保等价性**：`scale=shift=0` 时 `(1+0)*h+0 = h`，
   与原模型逐位相同，探针那套落点判据继续成立。
   梯度路径 `dL/dscale = dL/dh · h` 是内容相关的。
5. **per-latent-token 粒度**（§2.2）—— AdaLN 给不了，必须走独立通道；
   而现有 narrow/view/cat 的落点机制已经验证过 64/64 通过，可以直接复用。

### 关于 gate

Matrix-Game 的经验是"只在前一半 blocks 注入"。与其人工指定层数，
不如给每层一个**零初始化的可学习 gate**，让模型自己决定哪些层需要动作信息 ——
这也正是 H3 自身 AdaLN-Zero 的风格（`gate_msa` / `gate_mlp`）。

---

## 6. 验证方法：先验机制，再看效果

**不要等 8 分钟一条地渲视频看。** 改完先跑 500 步，用与 §1.3 相同的权重差分分析：

| 判据 | 当前（加性 bias） | 期望（FiLM） |
|---|---|---|
| 每步实际更新 / Adam 理论步长 | 1.92% | 明显更高 |
| ΔW 与 W 的方向余弦 | +0.064 | 明显更高（不再抵消） |
| `scale` 相对量级 | — | 应能在数百步内进入 1%~5% |

`code/abot/check_action_grad.py` 需要同步改：判据从"bias 列范数"变成"scale/shift 列范数"。
落点探针 `probe_action_injection.py` 也要改 —— 参考实现从 `index_add` 变成
`index_select + 乘加`，但"video 段连续、anchor 零污染、latent token 不错位"这三条判据不变。

机制活了再谈效果。

---

## 7. 不确定性（诚实边界）

FiLM 能解决**梯度抵消**和**量级锚点**这两条，有实测支撑。但**动作控制最终能不能到位**
还取决于一个架构分析回答不了的问题：

**8 维离散按键 + 单首帧条件，信息量是否足够让模型推断出画面该怎么变。**

LingbotWorld 用的是稠密的 **Plücker 相机位姿**——几何完备的信号，
每个 token 都知道自己对应的射线怎么动；Matrix-Game 的鼠标是**连续 pitch/yaw 增量**。
而按键是高度压缩的**间接**信号：从"按了 D"到"画面该怎么变"中间隔着朝向、地形、
相机跟随逻辑，这些都要模型自己从首帧和 prompt 推断。

这一点上，我们的设定比两个参考方案都更难。如果 FiLM 改完机制活了但控制力仍不足，
下一步应该考虑的不是继续换注入形式，而是**加强条件信号本身** ——
例如把相机增量也作为输入（但那在推理时不可得，见 `README.md` §6 的讨论），
或者退回到 Incantation 那种"文本描述动作"的路线，用 H3 那个 62GB text encoder
的语义能力去补上按键与画面之间的语义鸿沟。

---

## 附：本文所有数据的复现方式

```bash
conda activate minimax_h3
cd /opt/dlami/nvme/danze/minimax_finetune

# §1.1 注入量级        —— 读 video_patch_proj + stage-1 缓存 latent
# §1.3 梯度抵消        —— step-500 与 step-1000 权重差分
# §2.2 动作时间变化率  —— data/abot_meta_2000.jsonl 抽 300 条分箱后统计
```

参考资料：
- LingbotWorld-2：`/opt/dlami/nvme/danze/lingbot-world-v2/wan/modules/model_causal.py`
- Matrix-Game 2.0：https://arxiv.org/html/2508.13009v1
- Incantation：https://github.com/zhushangwen/Incantation
