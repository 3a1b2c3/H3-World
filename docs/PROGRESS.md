# MiniMax-H3 动作条件微调 · 进展记录

> 最后更新 2026-08-20 17:25

## 现在在哪

**第二次正式训练进行中** —— 换成 FiLM 注入方式，10 epochs / 9840 步，8 卡，预计 26 小时。

第一次正式训练（加性 bias，7872 条 × 3 epochs）已跑完，**结论是失败**：动作条件对生成
几乎没有影响。原因已经定位清楚，不是训练不够，是注入机制本身有缺陷（见 §4）。

```
训练      output/minimax_h3_abot/7872_film/     10 epochs × 984 步，SAVE_STEPS=500
速度      9.66 s/it（8 卡，85 GB/卡，满载）      与 bias 版 9.59 基本持平
起始 loss [0.038, 0.342]                        与 bias 版同期 [0.032, 0.277] 同分布
```

---

## 目标

让 **8 个按键**（W/A/S/D 移动 + I/J/K/L 视角）真正控制视频生成 ——
给定同一个首帧、不同的按键序列，应该生成出不同的、符合指令的画面运动。

任务设定是 `first_frame + 8d_actions`：只给首帧，不给末帧。

---

## 走到这一步的过程

### 一、数据与环境落地

- 从 COS 取回 `minimax_h3_bundle`（代码/文档/metadata）与 `abot_clips_8000`（38.5 GB 切片）
- 8000 条切片，每条 124 帧 @24fps / 832×480，配套逐帧动作 `[130, 17]`
- 环境：`DiffSynth-Studio-h3`（checkout 到补丁 base commit `300e3e4` 并打好
  `diffsynth_h3_action.patch`）、MiniMax-H3 FL2VA 权重 134 GB、conda 环境 `minimax_h3`
- 数据后来扩到 16000 条（`abot_meta_16000.jsonl`），当前训练用的仍是 8000 那批的
  7872/128 划分

**验证过的**：tier 前缀性质逐字节成立、8000 行 metadata 引用零缺失、
`abot_action.py` 自检全通过（`latent_t=37`、有效帧率 24.156）。

### 二、注入通路验证

stage 1 缓存 latent → 注入动作 → 落点探针，三步都过：

| 项 | 结果 |
|---|---|
| stage 1 冒烟 | 64/64，`latent_t=37`、anchor `(780,96)`、`seq_len=16640` 五项判据全中 |
| 动作注入 | 64 条，按键覆盖与相机 \|max\| 与 bundle 记录**逐项一致** |
| 落点探针 | **64/64 通过** —— 实现与参考逐位一致、bias 只落 video 行、anchor 零污染 |

**这一步的意义**：证明"动作张量确实被放到了序列里正确的位置"。这是训练验不出来的
——加错位置 loss 照样降，只是学到的东西没有物理意义。

### 三、动作可视化

`code/abot/viz_action.py` —— 把逐帧动作渲染成 HUD 叠在切片旁边，用于肉眼验证
**动作语义**（数值判据只能验"没错一帧"，验不了"符号对不对"）。

顺带验出一条此前只有推导没有实证的结论：

```
J − L 净帧数与累计 yaw 的符号，6 条样本全部一致
没按视角键的那条 yaw 仅 −4.7°（近零，天然阴性对照）
→ J = 左转 → yaw 正，L = 右转 → yaw 负
```

### 四、第一次正式训练（bias）与失效诊断

7872 条 × 3 epochs = 2952 步跑完，推理出来的结果"像在胡乱操控"。诊断如下。

**注入强度只有 hidden 的千分之一量级**：

| step | 单层 bias L2 | 相对 hidden(224.55) | 较上次 |
|---:|---:|---:|---:|
| 500 | 0.0947 | 0.042% | — |
| 1000 | 0.1521 | 0.068% | ×1.607 |
| 1500 | 0.1796 | 0.080% | ×1.180 |
| 2000 | 0.2157 | 0.096% | ×1.201 |
| 2500 | 0.2396 | 0.107% | ×1.111 |

增速持续衰减，训完也到不了有效区间（要 5% 调制量需要现在的 47 倍）。

**根因不是学习率，是梯度在互相抵消**：

```
每步实际更新 / Adam 理论步长    1.92%
ΔW 与已有 W 的方向余弦          +0.064   （几乎正交）
|W₅₀₀| + |ΔW| = 3.93           实际 |W₁₀₀₀| 只有 2.87
```

加性 bias 是 `Δx = b(action)`，**与画面内容无关**。但"按 D 键"的视觉效果依赖当前朝向和
场景——面朝东按 D 和面朝北按 D，画面变化完全不同。不同样本要求 b 指向不同方向，
平均下来就抵消了。**这是表达能力缺陷，调学习率只会让一个方向错误的量长得更快。**

### 五、方案调研与选型

调研了 8 种注入方式，结合 H3 架构的三条硬约束做了适配性评估
（完整分析见 `docs/action_injection_design.md`，可视版见 `docs/injection_options.html`）：

H3 的三条约束：
1. **没有 cross-attention** —— block 里只有 self-attn + mlp
2. **AdaLN 是粗粒度的** —— `combined_indices` 只有 O(10) 组，整个 video 段共享一组
   scale/shift；而动作在相邻 latent 步之间有 9.5% 的变化率，粗粒度会把时序抹掉
3. **RMSNorm + AdaLN-Zero** —— 调制接口在 norm 之后，而旧方案注入在 block 入口，位置错配

结论：能直接用的只有 LingbotWorld 的 **FiLM**；Matrix-Game 的 cross-attn 卡在约束 1，
Incantation 的文本接口卡在生成范式（它要自回归逐帧换 prompt，H3 是一次性全序列去噪）。

### 六、FiLM 改造（当前）

把注入从 block 入口的加性 bias，改成 **norm1+AdaLN 之后的 per-latent-token FiLM**：

```python
h = self.norm1(x)
h = _modulate_scale_shift(h, shift_msa, scale_msa, combined_indices)   # 原有 AdaLN
h = _apply_action_film(h, a_scale, a_shift, v0, n_video, latent_t, fr) # 新增
h = self.attn(h, ...)
```

旧路径完整保留（`--action_mode bias`）以便对照。改造前逐项验过：

| 检查 | 结果 |
|---|---|
| 零初始化与原模型逐位等价 | ✅ `(1+0)·h+0 = h` |
| anchor / pad 段未被触碰 | ✅ |
| 第 k 个 latent token 用的是 `scale[k]` | ✅ |
| 与 `index_select` 参考实现逐位一致 | ✅ |
| 单卡冒烟：动作层参数量 | ✅ 4,300,800 = 2×50×8×5376 |
| 3 步后梯度是否流到动作层 | ✅ `\|scale\|max 9.35e-05`，非零 |
| checkpoint 格式 | ✅ `action_scale`×50 + `action_shift`×50 + LoRA 208 |
| 推理侧能读新格式 | ✅ `mode=film` |
| 旧 bias checkpoint 向后兼容 | ✅ 仍识别为 `mode=bias` |

> 改造中发现并修掉一个会让 26 小时白费的坑：`infer_abot.py` 原先硬编码只认
> `action_embedders.` 前缀，FiLM 的 checkpoint 推理时会直接 raise。

---

## 下一步：先验机制，再看效果

**不要等 26 小时训完再渲视频判断。** 第一个 checkpoint（500 步，约 80 分钟）出来后：

```bash
conda activate minimax_h3
python3 code/abot/compare_action_ckpt.py --dir output/minimax_h3_abot/7872_film
```

工具会直接打结论行。判据（括号是 bias 版的失效基线，工具已用旧 checkpoint 校准过
能准确复现这两个数）：

| 判据 | 失效基线 | 期望 |
|---|---|---|
| ΔW 与 W 的方向余弦 | +0.064 | 明显更高（> +0.4 为好） |
| 每步实际 / Adam 理论步长 | 1.92% | 明显更高 |
| `\|scale\|` 平均值 | — | 数百步内进入 1%–5% |

**分支决策**：如果到 1000 步余弦仍在 +0.1 以下，说明换机制也没解决，
不必跑满 26 小时 —— 那时该走的是下面这条路，而不是继续等或者再换注入形式。

### 备选路线：换信号表示（很可能是必经之路）

调研 Hunyuan-GameCraft 时发现的一层，之前的分析漏了：**"注入什么"可能比"怎么注入"更关键。**
它把键盘鼠标统一映射到**共享的相机表示空间**再注入，把"按了 D"到"画面该怎么变"
中间的朝向、地形、相机跟随逻辑显式补上。

这条对我们特别可行，因为**数据里现成就有**：切片 npy 是 17 维，8 个按键之外还有 6 个
COLMAP 反推的相机通道，稠密（33/37 token 非零）、几何意义明确。
之前否掉是因为推理时相机增量不可得，但 GameCraft 给了出路 ——
训练时用相机表示做条件，另训一个「按键 → 相机增量」的轻量映射供推理时用。
实测相机通道能以 **AUC 0.88–0.99** 反向预测按键，说明这个映射强相关、可学。

**为什么说很可能是必经之路**：按键是高度压缩的**间接**信号，而 LingbotWorld 用的是
几何完备的 Plücker 位姿、Matrix-Game 用的是连续 pitch/yaw 增量。这一点上我们的设定
比两个参考都更难，FiLM 只解决了注入侧的问题，没解决信号侧的。

---

## 已知的其他问题

**cam-dropout 给 62.5% 的样本贴了错误标签**（当前 8 维方案已不用相机通道，
但若走上面的备选路线会重新遇到）：置零 ≠ 缺失，零向量在数值上就是"相机静止"。
64 条冒烟里被置零的 40 条**没有一条**接近静止（运动 |max| 中位 1.83）。
建议加第 18 维做 `cam_valid` 标志位。

**推理侧的两个遗留**（`docs/action_injection_design.md` §6.1）：
`action_cond` 在 CFG 正负两侧共用，动作不参与 CFG，控制力弱时没有旋钮可放大。

---

## 产物索引

| 路径 | 内容 |
|---|---|
| `README.md` | 目录布局、环境、起训命令、缓存落盘规则 |
| `docs/action_injection_design.md` | 注入方案的完整技术分析与推导 |
| `docs/injection_options.html` | 上文的可视版（框图 + 判定 + 计划） |
| `code/abot/compare_action_ckpt.py` | **机制诊断工具**，两个 checkpoint 一比看死活 |
| `code/abot/viz_action.py` | 动作 HUD 渲染（单条切片） |
| `code/abot/build_compare_page.py` | GT/Generated 横版对比页 |
| `code/abot/infer_abot.py` | 推理（支持 film / bias 两种 checkpoint） |
| `output/abot_inference/step2952_8samples/` | bias 版 step-2952 的 8 组推理结果 |
| `output/minimax_h3_abot/7872/` | 第一次训练（bias）的 6 个 checkpoint |
| `output/minimax_h3_abot/7872_film/` | **当前训练**（FiLM） |
