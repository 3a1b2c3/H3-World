# ABot 数据线：进展总结

> 2026-08-18
> 详细设计与实测依据见 [`minimax_h3_abot_data.md`](minimax_h3_abot_data.md)
> 本文只回答三件事：**做了什么、怎么做的、接下来做什么。**

---

## 一句话

把 ABot-World-Explorer-500h（30969 条 60s 开放世界探索录像）处理成了 MiniMax-H3
可训练的格式，8000 条切片已就位，64 条冒烟的 latent 缓存与动作注入全部跑通并验证。
**模型代码一行没改** —— 复用 V-Rising 那轮已验证的 Stage B 动作注入通路。

---

## 一、做了什么

| 阶段 | 状态 | 产出 |
|---|---|---|
| 数据勘察 | ✅ | 摸清格式、发现 `delta_*` 全零、确认 COLMAP 位姿可用 |
| 切片管线 | ✅ | 8000 条 832×480 @24fps 切片 + 逐帧动作 npy，37 GB |
| metadata | ✅ | 前缀嵌套的 64 / 2000 / 8000 三档 |
| 冒烟 stage 1 | ✅ | 64 条 latent 缓存，0.86 GB，12分41秒 |
| 动作注入 | ✅ | 17 通道注入 64 条缓存 |
| 落点探针 | ✅ | 64/64 通过 |
| **冒烟 stage 2** | 🔴 **被显存阻塞** | 两次 OOM，非技术问题，见 §5.0 |
| 正式训练 | ⬜ | 2000 条 × 3 epoch |

---

## 二、怎么做的

### 2.1 数据勘察：三个关键发现

**① `action.json` 的连续动作字段全是 0。**
README 郑重列出的 `delta_translation_cam` / `delta_euler_deg` 及 `_smooth` 版本，
实测 296 条 episode **没有任何一帧非零**。所以连续运动信号必须另找来源。

**② COLMAP 位姿可以顶上，但尺度不能信。**
`sparse/0/images.txt` 逐帧位姿，覆盖率 **1800/1800 = 100%**。
反推的按键→运动因果表非常干净（W/S 走 ±dz、A/D 走 ±dx、I/K 走 ±pitch、J/L 走 ±yaw，
符号成对相反）。但同样按 W 前进，**逐 episode 尺度差 33 倍**
⇒ 平移通道必须按 episode 归一化，旋转是角度不受影响。

**③ 帧率关系比 V-Rising 有利。**
V-Rising 是 16→24 **上采样**（32.7% 重复帧、尾部丢 0.56s）；
这份是 30→24 **下采样**，两个硬伤都不存在。

### 2.2 切片：把对齐的决定权拿回自己手里

核心设计是**不去复刻 H3 内部的 `round()` 帧映射**，而是自己定一条
ffmpeg 能精确表达、也能精确求逆的规则：**每 5 个源帧丢第 5 个**（30×4/5=24 精确成立）。

```
输出帧 j:  0  1  2  3 | 4  5  6  7 | 8  9 10 11
源帧偏移:  0  1  2  3 | 5  6  7  8 | 10 11 12 13
                    ↑丢4         ↑丢9
```

**同一个帧号列表同时用来裁视频和取动作** ⇒ 对齐是构造性的，不依赖对框架内部行为的推断。
切片直接产成 832×480 @24fps，于是 stage 1 里的重采样和缩放都退化成恒等。

三件事顺带一次做完：降帧率、降分辨率、动作对齐。

### 2.3 关键参数

| 参数 | 值 | 理由 |
|---|---|---|
| `num_frames` | **124** (=17×7+5) | H3 官方 example 清一色 124；V-Rising 用 107 只因源片太短 |
| `latent_t` | **37** | 已由真实 VAE 输出证实 |
| 分辨率 | 832×480 | 训练分辨率，切片时一次到位 |
| 写盘帧数 | 130 (124+6) | 余量，防 `floor(duration×24)` 浮点误差把帧数降级回 107 |
| prompt | `scene_static` | **不用** `narrative` —— 它描述整条 60s 的运动，与 5.17s 窗口不符，会和动作条件打架 |
| 动作通道 | **17** | 11 键 + 3 旋转增量 + 3 平移增量 |

### 2.4 行序纪律

所有 tier 都是同一 manifest 的**字节级前缀**（64 ⊂ 2000 ⊂ 8000），
md5 逐档核对通过。这样小 tier 的 latent 缓存能被大 tier 直接复用
（姊妹文档 §7.4b 的教训，这次从设计之初就按它做）。

---

## 三、验证了什么

所有结论都有可复现的判据，不是"跑通了就算数"。

| 验证项 | 判据 | 结果 |
|---|---|---|
| 视频/动作逐帧对齐 | 位移扫描 argmin 必须落在 0 | **18/18 PASS**，干净 V 形 |
| 无重复帧 | 逐位比较相邻帧 | **0/124**（V-Rising 是 35/107） |
| 124 帧未被降级 | `input_latents.shape[2] == 37` | ✅ 37 |
| 时间轴推导成立 | 真实 VAE 输出 vs `frame_spans` 累加 | ✅ 对撞通过 |
| 前缀保序 | 字节级 + md5 | ✅ 三对全通过 |
| 注入落点 | `narrow` 实现与 `index_add` 参考逐位一致；anchor / text / audio 零污染 | **64/64 PASS** |

### 实测数字

```
切片      8000 条 / 32 min / 0.24 s/片 / 零失败 / 37 GB
stage 1   64 条 / 12分41秒 / 11.9 s/clip / 0.86 GB (13.48 MB/条)
seq_len   16640  (text 964 + anchor 780 + audio 414 + video 14430)
          对比 V-Rising 14464
```

---

## 四、过程中修正的四个判断

记下来是因为它们都是"小样本上看起来没问题、大样本才暴露"的类型。

1. **旋转也有重尾。** 64 条上 `d_yaw` max 仅 1.41，看着干净；2225 条上是 **21.15**（p99 才 1.18）。
   加了 `ROT_CLIP=3.0`。判据不只是分位数拐点，更强的是
   **`yaw≥3.0` 的极值 token 只有 15% 同时按着转向键** ⇒ 是位姿重建跳变而非真实运动。
2. **episode 帧数不都是 1800。** 40 条抽样得出"统一 60.0s"，8000 条上见到 2205 / 2281 / 2348 帧的。
   代码本来就按 `total_frames` 动态算，没出错，但文档结论过窄。
3. **窗口放置有偏。** 第一版所有 w=0 窗口都落在源帧 0~80（镜头刚起步、动作稀疏）。
   改成按 `sample_id` 播种打乱槽位，窗口 0 均匀散布在整条 60s 上。
4. **缓存体积估算，我先低估又高估。** 大头是 `prompt_embeds` 不是 latent；
   但 text rows 是**次线性**增长的（114 词 → 964 行，不是按词数比例的 1220），
   最终实测 13.48 MB/条，接近最初估的 14 MB。

还有一条**不是** bug：`cam-dropout` 在 64 条上置零了 62.5% 而非 50%，
但在 2000/8000 档上都是 50.5% —— 小样本噪声，代码未动。

---

## 五、接下来做什么

### 5.0 🔴 当前阻塞：没有卡（2026-08-19）

冒烟 stage 2 试了两次，都因**邻居作业膨胀**而 OOM。**不是配置问题** ——
我们这侧一次比一次省，是对方在涨：

| 尝试 | 我方峰值 | 崩在哪 | 当时邻居 |
|---|---:|---|---:|
| `--fp8_models dit` | 33.6 GB | 加载权重 | 94 → **106 GB** |
| 上面 + `--use_gradient_checkpointing_offload --enable_model_cpu_offload` | **22.1 GB** | **反向传播** | 106 → **117 GB** |

第二次已经跑到 backward，说明 offload 通路是通的、`action_embedders` 与
`--enable_model_cpu_offload` 没有冲突（这曾是个担心点）。纯粹是卡被吃光：
GPU4 一度到 142613/143771 MiB，**一个邻居进程占 117 GB**。

八张卡当时都被两个多卡作业占着（`yinhanzhang` 的 Wan2.2 四卡 ×94 GB、
`zeqingwang` 的 ReactiveGWM），最大空闲 48 GB 且在波动。
这与姊妹文档 §6.4 / §9.3 的判断一致：**资源协调问题，不是技术问题**，
错峰比任何技术手段都有效。

**已确认无残留**：我方无进程、不占显存、无半截权重。

### 恢复方法：等到有 ≥ 55 GB 空闲的卡，直接跑

```bash
# 干净的卡（推荐，bf16，数值无损）
cd /nfs/danze/repo/DiffSynth-Studio-new
CUDA_VISIBLE_DEVICES=<N> STAGE=2 ACTION_BUTTONS=17 ACTION_TRAIN_ONLY=1 \
  SAVE_STEPS=50 NUM_EPOCHS=5 \
  bash examples/minimax_h3/model_training/lora/ABot-FL2VA.sh 64

# 卡不够干净时（脚本已支持 EXTRA_ARGS 透传）
EXTRA_ARGS="--fp8_models dit --use_gradient_checkpointing_offload" ...    # 需 ~40 GB
EXTRA_ARGS="... --enable_model_cpu_offload" ...                            # 需 ~25 GB，但慢
```

⚠️ **fp8 只适合做通路诊断，不适合评估效果** —— 它改变权重数值。
诊断本身对 fp8 稳健：`dW = dYᵀ·x`，输入恒零的通道梯度精确为 0，与精度无关。

### 下一步（有卡之后）：冒烟 stage 2（约 2 小时）

```bash
cd /nfs/danze/repo/DiffSynth-Studio-new
CUDA_VISIBLE_DEVICES=2 STAGE=2 ACTION_BUTTONS=17 ACTION_TRAIN_ONLY=1 \
  SAVE_STEPS=50 NUM_EPOCHS=5 \
  bash examples/minimax_h3/model_training/lora/ABot-FL2VA.sh 64
```

要回答两个问题，都靠**存盘权重的逐通道梯度**（不看 loss 曲线 —— flow matching 下没有判据性）：

| 问题 | 判据 |
|---|---|
| 17 通道语义有没有串列 | 梯度非零的通道集合必须**精确等于**数据里出现过的通道。Q/E/Space 应当**精确为 0**（天然阴性对照），其余 14 个非零 |
| **有没有捷径学习** | 比按键通道与相机通道的 `\|w\|max`。按键明显偏低 ⇒ 模型在偷懒读相机、按键通路没训练 |

第二条是本方案最大的未知数。相机增量是动作的**结果**，信息量远大于按键，
模型可能干脆只读它。已用 `--cam-dropout 0.5` 预防，但比例是拍的，要看诊断结果调。

### 然后：正式训练

**2000 条 × 3 epoch = 6000 步**。依据：rank-32 LoRA 63M 参数，1000–10000 步收敛；
同样步数下宁可多数据少 epoch（V-Rising 那轮 1374 条跑 3.6 epoch 出现过疑似过训的负面信号）。

成本：stage 1 约 6.9 h / 27 GB，stage 2 约 33 h（按当前争抢状态估）。
效果不够再上 8000 档 —— 因为前缀保序，2000 的缓存可直接复用，只需增量编码 6000 条。

### 仍然开着的风险

1. **捷径学习**（见上）—— 最大未知数。
2. **第三人称视角**。255/256 是 third person，画面里有玩家角色，
   动作同时驱动角色和相机，比纯相机控制多一层要学。
3. **动作是否真的在控制生成** —— 需要同首尾帧、同 seed、喂不同动作序列看画面差异。
   这是"Stage B 跑通"与"Stage B 对了"之间剩下的最后一段，冒烟验不到。
4. **是否该上更长的 `num_frames`**（141/175）。数据不再有时长约束，
   唯一代价是算力。Stage C（history-conditioned rollout）时重算这笔账。

---

## 六、文件索引

| 路径 | 说明 |
|---|---|
| `/nfs/danze/abot/abot_action.py` | 动作 schema + COLMAP 反推 + 非均匀分箱（带 self-test） |
| `/nfs/danze/abot/build_abot_clips.py` | 切片 + metadata + 动作 npy；`--verify` 做对齐抽查 |
| `/nfs/danze/abot/inject_abot_action.py` | 缓存事后注入（`--cam-dropout` / `--keys-only`） |
| `.../lora/ABot-FL2VA.sh` | 训练入口（STAGE / ACTION_BUTTONS / NUM_EPOCHS 可配） |
| `/nfs/danze/data/abot/` | 切片、动作 npy、manifest 与三档 metadata |
| `/nfs/danze/model/minimax_h3_abot/64-cache/` | 冒烟 latent 缓存（已注入动作） |
| `/nfs/danze/logs/abot_*.log` | 切片与 stage 1 日志 |
| `/nfs/danze/probe_action_injection.py` | 落点探针（复用，无需改动） |
