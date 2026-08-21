# ABot-World-Explorer-500h → MiniMax-H3：数据处理方案

> 更新：2026-08-17
> 源数据：`/nfs/yixinyang/code/LongLive/data/ABot-World-Explorer-500h`（2.5 TB，只读）
> 产出：`/nfs/danze/data/abot/`　代码：`/nfs/danze/abot/`
> 姊妹文档：[`minimax_h3_architecture_and_data.md`](minimax_h3_architecture_and_data.md)（架构、数据契约、帧率重采样）
> 　　　　　[`minimax_h3_vrising_finetune.md`](minimax_h3_vrising_finetune.md)（V-Rising 数据选型与运维决策）
> 　　　　　[`minimax_h3_world_model.md`](minimax_h3_world_model.md)（动作条件世界模型改造）
> 　　　　　[`minimax_h3_abot_progress.md`](minimax_h3_abot_progress.md)（**进展总结**：做了什么/怎么做/下一步）

本文回答：**这份新数据长什么样、怎么变成 H3 能吃的格式、动作怎么注入。**

模型侧一行没改 —— 复用 Stage B 已经验证过的那套注入通路（世界模型文档 §4/§7.0）。
本文全部工作在数据侧。

> 🔴 **一句话结论**：这份数据比 V-Rising **更适合**做世界模型。
> 帧率关系从「16→24 上采样」变成「30→24 下采样」，
> [架构文档](minimax_h3_architecture_and_data.md) §5 记的两个硬伤
> （32.7% 重复帧、尾部丢 0.56s）**在这里都不存在**，已实测。
> 而且逐帧 COLMAP 位姿给了连续的相机运动信号，V-Rising 完全没有。

---

## 0. 速查

| 问题 | 一句话答案 | 详见 |
|---|---|---|
| 源数据是什么 | 30969 条 60s / 1800 帧 / 1920×1080 / 30fps / 无音轨的第三人称探索录像 | §1 |
| 有没有动作 | **有**，11 键逐帧 + 逐帧 COLMAP 位姿（覆盖率 100%） | §2 |
| `action.json` 的 delta_* 能用吗 | ❌ **全是 0**，296 条抽样无一例外，必须自己从 COLMAP 反推 | §2.2 |
| 切多长 | **124 帧 @24fps = 5.167s**，H3 原生值 | §3.1 |
| 帧率怎么降 | 每 5 个源帧丢第 5 个，切片直接就是 24fps，`LoadVideo` 恒等读回 | §3.2 |
| 有重复帧吗 | **0**（V-Rising 是 35/107） | §3.4 |
| 对齐验了吗 | ✅ 位移扫描 argmin=0，**18/18 通过**（8 条 + 全量后 10 条） | §3.4 |
| 动作怎么注入 | 17 通道（11 键 + 3 旋转 + 3 平移），走缓存事后注入 | §4 |
| prompt 用哪个字段 | `scene_static`，**不用** `narrative` | §3.5 |

---

## 1. 源数据实测

全部数字来自实测，不是 README 的宣称值。README 自己也声明
「`500h` 是标识符不是审计过的时长」。

| 项 | 值 | 怎么测的 |
|---|---|---|
| episode 数 | **30969** | `find data -name video.mp4 \| wc -l`，与 metadata.jsonl 行数一致 |
| 磁盘 | **2.5 TB** | `du -sh data` |
| 分辨率 | 1920×1080 | ffprobe，40 条抽样全一致 |
| 帧率 | **30fps** | ffprobe，全一致 |
| 帧数 / 时长 | **多数 1800 / 60.0s，但不是全部** | 40 条抽样里 39×1800 + 1×1837；**8000 条切片上见到 2205 / 2281 / 2348 帧的** ⇒ 不要写死 1800 |
| 音轨 | **无** | ffprobe 只有一条 h264 video stream |
| 相机 | PINHOLE 1920×1080 f=1099.15 cx=960 cy=540 | `sparse/0/cameras.txt` |
| 视角 | **third 255 / first 1**（256 条抽样） | `caption.json.perspective` |
| prompt 长度 | 中位 **114** 词（49–179，8000 条实测） | `caption.json.scene_static` |
| COLMAP 位姿覆盖 | **1800/1800 = 100%**，40 条抽样全满 | §2.3 |
| `points3D.txt` | **空文件** | 见 §2.3 的推论 |

一条 episode 的目录就两个文件：

```
data/<prefix>/<sample_id>/video.mp4          ~111 MB
data/<prefix>/<sample_id>/annotations.tar    ~1.3 MB
```

`annotations.tar` 是未压缩 USTAR，固定 5 个成员：
`action.json` / `caption.json` / `sparse/0/{cameras,images,points3D}.txt`。
**读之前要校验成员白名单和文件类型**（skill 文档的硬性要求，`abot_action.read_episode`
里已经做了），不要 `extractall()`。

### 1.1 与 V-Rising 的对照

| | V-Rising | ABot | 影响 |
|---|---|---|---|
| clip 数 | 20699 | 30969 episodes | 切片后可远超 |
| 分辨率 | 832×480（就是训练分辨率） | 1920×1080 | 要缩，见 §3.3 |
| 帧率 | 16fps → **上采样**到 24 | 30fps → **下采样**到 24 | 🎉 关键改善，见 §3.4 |
| 单条时长 | 5.06s（不够 124 帧） | 60s | 可自由选 num_frames |
| 音轨 | 无 | 无 | 同样靠 `--silent_on_missing_audio` 兜底 |
| 动作 | 12 键二值 | 11 键二值 **+ 逐帧 6DoF 位姿** | 🎉 多了连续信号 |
| 域 | 单一游戏、窄切片 | 开放世界探索、场景极多样 | 泛化更好，但更难拟合 |

---

## 2. 动作数据：有，但不在你以为的地方

### 2.1 `action.json` 的按键是好的

```json
{"control_scheme": "WASD_QE_locomotion_IJKL_rotation",
 "fps": 30.0, "original_fps": 30.0, "sample_stride": 1, "total_frames": 1800,
 "start_frame_index": 12600, "end_frame_index": 14400, "thresholds": {},
 "frames": [{"frame_id": "frame_000001", "timestamp": 0.0,
             "keys": {"W": false, ..., "J": true, ...},
             "delta_translation_cam": [0,0,0], "delta_euler_deg": [0,0,0],
             "delta_translation_cam_smooth": [0,0,0], "delta_euler_deg_smooth": [0,0,0]}]}
```

11 个键：`W A S D Q E I J K L Space`。40 条抽样的按下率：

| 键 | 按下率 | 有该键的 episode |
|---|---:|---:|
| W / A / S / D | 22.4% / 20.5% / 22.1% / 23.0% | 40/40 |
| J / L（左右转） | 28.3% / 28.0% | 40/40 |
| I / K（上下看） | 10.9% / 10.6% | 40/40 |
| **Q / E / Space** | **0.00%** | **0/40** |

**8 个键活着，3 个恒零。** 比 V-Rising 的「12 键里 5 键活着」还密集，
而且没有稀疏事件键 —— 全是持续按住的移动/转向键。对世界模型很友好。

> ⚠️ Q/E/Space 的恒零已在 **2225 条实际切片**上复核（§4.1），但全集 30969 未扫，
> 仍不宜断言。`abot_action.py` 保留全部 11 通道，代价只是 3×5376×50 个
> 拿不到梯度的参数（与 V-Rising §3.5 同一个取舍），而且它们正好当
> §4.2 梯度诊断的**阴性对照**。

### 2.2 🔴 `delta_*` 四个向量全是 0 —— 别信 schema，要实测

README 把 `delta_translation_cam` / `delta_euler_deg` 及其 `_smooth` 版本
列成正式字段，看起来正是我们要的连续动作信号。**实测 296 条 episode，
没有任何一条、任何一帧、任何一个分量非零。**

```
episodes scanned: 256          （另有 40 条在别的抽样里）
episodes with ANY nonzero delta_* vector: 0
```

**所以连续运动信号必须自己从 COLMAP 位姿反推。** 这不是绕路 ——
反推出来的质量更高（是位姿本身，不是被某个未公开阈值处理过的量），
而且 README 本来就警告过「`action.json` 的 deltas 是相对动作标注，
不等同于 COLMAP 绝对位姿，未经验证的换算不要混用」。既然一边是空的，
就直接用另一边，不存在混用问题。

### 2.3 COLMAP 位姿：满覆盖，但尺度逐 episode 任意

`sparse/0/images.txt` 每帧一条记录，`NAME` 与 `action.json` 的 `frame_id`
逐条对得上（40/40 验过），**覆盖率 1800/1800**。

```
1 0.3718627522 0.4612331639 0.6323212114 -0.4991512272 -1074.047 481.268 -1608.786 1 frame_000001.jpg
```

按 COLMAP 约定：这是 **world→camera**，`X_cam = R_cw X_world + t_cw`，
相机中心 `C = -R_cw^T t_cw`。`TX TY TZ` **不是**相机位置。

> 📌 **`points3D.txt` 是空的** —— 一个没有任何稀疏点的「重建」。
> 合理推测是引擎导出的真值位姿套了 COLMAP 的文本格式，而不是真跑了 SfM。
> 这解释了为什么覆盖率能到 100%（真 SfM 做不到），
> 但**不能**据此认为尺度是统一的 —— 见下。

**尺度实测（40 条 episode）**：

| 量 | min | median | max | max/min |
|---|---:|---:|---:|---:|
| 每帧位移中位数 | 0.0140 | 0.0427 | 0.0793 | **5.7×** |
| 只按 W 时的每帧位移 | 0.0045 | 0.0523 | 0.1488 | **32.9×** |

同样是「按住 W 往前走」，不同 episode 的数值差 33 倍。
**所以平移通道必须按 episode 归一化**（`abot_action.episode_translation_scale`，
除以该条 episode 抽帧后的每帧位移中位数）。
旋转是角度、天生无尺度问题，直接用。

### 2.4 按键 → 相机运动的因果核对

这一步不是走过场：它同时验证了「键的语义」「位姿的坐标约定」「两者的时间对齐」。
6 条 episode，对每个键算「按下时的平均增量 − 未按下时的平均增量」：

| 键 | dx(右) | dy(下) | dz(前) | pitch | yaw | roll | 读出来的语义 |
|---|---:|---:|---:|---:|---:|---:|---|
| W | +0.002 | −0.006 | **+0.034** | −0.009 | −0.004 | −0.005 | 前进 |
| S | −0.001 | +0.008 | **−0.031** | −0.017 | +0.122 | +0.034 | 后退 |
| D | **+0.033** | +0.002 | −0.009 | +0.004 | −0.128 | −0.014 | 右平移 |
| A | **−0.018** | +0.001 | +0.004 | −0.031 | +0.306 | +0.051 | 左平移 |
| I | −0.001 | +0.038 | −0.003 | **−0.438** | +0.001 | −0.061 | 抬头 |
| K | −0.002 | −0.038 | +0.001 | **+0.449** | −0.027 | +0.048 | 低头 |
| J | +0.050 | −0.002 | −0.002 | +0.010 | **+0.742** | +0.114 | 左转 |
| L | −0.055 | +0.003 | −0.009 | −0.023 | **−0.744** | −0.124 | 右转 |

八个键的主分量各归各位，符号成对相反（W/S、A/D、I/K、J/L），
量级也对称。**坐标约定和时间对齐都没问题。**

> ⚠️ A 和 S 的 yaw 分量偏大（+0.306 / +0.122），这是**共现**造成的：
> 玩家常一边平移一边转向。这里算的是边际差不是偏效应，不能读成「A 会转向」。
> 不影响用途 —— 我们要的是把两种信号都喂给模型，不是做因果推断。

---

## 3. 切片方案

### 3.1 为什么是 124 帧

`--num_frames` 必须是 `17n+5`。V-Rising 被迫用 107，是因为源 clip 只有
5.06s、24fps 下最多 121 帧（架构文档 §5.5）。**这里源是 60s，约束不存在了**，
所以应当直接选 H3 的原生值：

| num_frames | latent_t | 时长 | video rows | 判断 |
|---|---:|---:|---:|---|
| 107 = 17×6+5 | 32 | 4.458s | 12480 | V-Rising 的值，无理由沿用 |
| **124 = 17×7+5** | **37** | **5.167s** | **14430** | ✅ **选它** |
| 141 = 17×8+5 | 42 | 5.875s | 16380 | 更长，算力 +13% |
| 175 = 17×10+5 | 52 | 7.292s | 20280 | Retake example 用过 |
| 192 = 17×11+5 | 57 | 8.000s | 22230 | 算力 +54%，Stage C 再说 |

**124 是 H3 全部官方 example 的默认值**（`model_inference/*.py` 清一色
`num_frames=124`，`MiniMaxH3ReferenceLoader` 的默认形参也是 124）。
选它 = 最贴预训练的运动先验，这与架构文档 §5.6「不要随便偏离 24fps 先验」
是同一条理由的正面应用。

`latent_t = ((124−5)//17)×5+2 = 37`，且分组累加 `7×17+(1+4) = 124` ✓
（`abot_action.py` 的 self-test 对 5 个候选值都验了）。

### 3.2 30fps → 24fps：每 5 帧丢 1 帧

**不复刻 H3 内部的 `round()` 映射，而是自己定一条能被 ffmpeg 精确表达、
也能精确求逆的规则**：丢掉每 5 个源帧里的第 5 个。

```
输出帧 j:  0  1  2  3 | 4  5  6  7 | 8  9 10 11
源帧偏移:  0  1  2  3 | 5  6  7  8 | 10 11 12 13
                    ↑ 丢 4        ↑ 丢 9
```

`30 × 4/5 = 24` 精确成立。ffmpeg 侧就是一句
`select='not(eq(mod(n-S,5),4))'`，Python 侧就是
`offset(j) = (j//4)*5 + j%4`。**同一个列表同时用来裁视频和取动作**，
于是对齐是构造性的，不依赖对 `core/data/operators.py` 内部行为的推断。

> 这是与 V-Rising 那条路线最重要的方法论差别。那边是「读 H3 源码，
> 在 `h3_action.py` 里复刻 `map_single_frame_id`」（世界模型文档 §1 差异 3b），
> 对了，但**是推断出来的、且曾经漏过一次**。这边是「把决定权拿回自己手里」。

### 3.3 1920×1080 → 832×480

在 ffmpeg 里一次做完：`scale=-2:480,crop=832:480`。
1920×1080 缩到 854×480 再中心裁掉左右各 11px（约 2.5% 宽度）。

好处是**切片本身就是训练分辨率**，stage 1 里的 `ImageCropAndResize`
退化成恒等（scale=max(832/832, 480/480)=1），不会二次重采样。

### 3.4 ✅ 实测：0 重复帧、不丢尾帧、对齐 argmin=0

写盘 **130 帧**（124 + 6 帧余量），训练取前 124。留余量是为了
`get_available_num_frames` 用 `floor(duration × 24)` 时不会因浮点误差
掉到 124 以下、进而被 `get_num_frames` 一路降级到 107。

`LoadVideo(num_frames=124, frame_rate=24, fix_frame_rate=True)` 读回来：

```
LoadVideo returned 124 frames (asked 124)   size=(832, 480)
consecutive duplicate frames: 0/124          ← V-Rising 是 35/107 = 32.7%
```

对齐判据用**位移扫描**而不是绝对像素差（绝对差有 5~6/255 的重采样地板噪声，
慢镜头下和「错一帧」差不多大，没有分辨力；但地板噪声对所有 shift 是共同的，
所以 argmin 仍然干净）：

```
sample          nfrm  dup  shift-2  shift-1  shift+0  shift+1  shift+2   argmin
55ddfc809f3e     124    0    21.48    19.46     9.11    13.92    20.86     +0
92168909e1b2     124    0    23.52    20.18     7.62    18.02    22.50     +0
13940e1a5231     124    0    24.33    21.51     6.20    16.11    23.47     +0
...
8/8 通过（124 帧 / 0 重复帧 / 位移扫描 argmin=0）
```

八条全是以 shift=0 为底的干净 V 形。

> 🎉 **这条直接关掉了世界模型文档 §7.0d**（「分箱+重采样的时序对齐仍未验」，
> 那边标为「Stage B 剩余最高风险」）。在 ABot 这条数据线上它不再是风险 ——
> 视频与动作用同一个帧号列表取，且已实测复核。
> V-Rising 那条线上它依然开着。

### 3.5 prompt 用 `scene_static`，不用 `narrative`

`caption.json` 四个字段：

| 字段 | 中位词数 | 内容 |
|---|---:|---|
| `narrative` | 158 | 场景 **+ 整条 60s 的镜头运动**叙述 |
| `scene_static` | 114 | 纯静态环境描述 |
| `dense_temporal` | — | **空数组**，256/256 全空 |
| `perspective` | — | `third`（255/256） |

`dense_temporal` 本来是最理想的（分段时序描述），**但全空**，用不了。

于是在 `narrative` 和 `scene_static` 之间选。**选 `scene_static`**，理由是
决定性的：`narrative` 描述的是**整条 60s 的运动**，而我们只切 5.17s 的窗口，
它描述的镜头运动在这个窗口里**多半没发生**。把「镜头向左漂移」写进 prompt、
而动作通道说的是「按住 L 右转」，等于让文本条件和动作条件互相打架 ——
这正好破坏我们唯一真正想要的东西。

**运动信息应当只从动作通道来，文本只负责「这是什么场景」。**
`scene_static` 恰好就是窗口无关的，对同一条 episode 的任何窗口都成立。

> `narrative` 仍然写进了 metadata（不参与训练），留着备用 ——
> 比如以后想做「文本描述运动」的对照实验，或者拿它去蒸馏分段 caption。

---

## 4. 动作注入方案

### 4.1 17 通道

```
 0..10   11 个按键        二值   窗口内 amax
11..13   旋转增量 pitch/yaw/roll   度   窗口内 sum，再 / 4.0
14..16   平移增量 x右/y下/z前      已按 episode 归一化   窗口内 sum，再 / 4.0，截断 ±4
```

**二值用 amax、连续用 sum**：按键是脉冲，窗口内按下过就算按下，取均值会稀释成
0.25 这种没有物理含义的强度（沿用 V-Rising 的结论）；而增量是可加物理量，
一个覆盖 4 帧的 latent token 就该拿到这 4 帧的**总位移**。

H3 的分组是非均匀的 `(1,4,4,4,4)`，每组第一个 token 只覆盖 1 帧，
所以它的 sum 天然只有后面的 1/4。**这是真实信息不是假象** —— 那个 token
确实只代表 1 帧的时间。

**2225 条切片**上的实测分布（分箱缩放后、截断前）：

| 通道 | 活跃 clip | p95 | p99 | p99.9 | \|max\| |
|---|---:|---:|---:|---:|---:|
| W/A/S/D | 992/874/918/977 | 1.00 | 1.00 | 1.00 | 1.00 |
| I/J/K/L | 717/1294/720/1258 | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q/E/Space** | **0/2225** | 0 | 0 | 0 | **0.00** |
| d_pitch | 2062 | 0.71 | 0.74 | 0.97 | 3.16 |
| **d_yaw** | 2066 | 1.07 | 1.18 | 1.70 | **21.15** |
| d_roll | 2064 | 0.26 | 0.49 | 0.79 | 2.29 |
| d_x / d_y / d_z | 2080 / 2080 / 2081 | 1.98 / 1.09 / 1.20 | 2.43 / 1.48 / 2.27 | — | **47.5**(未截断) |

### 🔴 两个连续段都必须截断（`TRA_CLIP=4.0` / `ROT_CLIP=3.0`）

只在 64 条上看时旋转的 max 才 1.41，**像是干净的**；到 2225 条才暴露出
`d_yaw` 能到 21.15。这是「小样本上定的常数要在大样本上复核」的一个实例。

阈值不是拍的，两条依据：

1. **分位数有明显拐点**：yaw 的 p99.9 = 1.70，p99.99 = 9.59。
   连续长尾不会这样跳，这是离散离群点。
2. **极值和按键对不上**：`yaw ≥ 3.0` 的 48 个 token 里，
   **只有 15% 同时按着 J 或 L**。真的快速转身不可能不按转向键 ——
   所以这些是位姿重建跳变，不是真实相机运动。

截断代价（2225 条上实测）：

| | 阈值 | 切掉的元素 | 受影响 clip |
|---|---:|---:|---:|
| 旋转 | 3.0（=12 度/箱 = 72 度/秒） | 0.020% | 10 / 2225 |
| 平移 | 4.0 | 0.174% | — |

72 度/秒以内的真实快速转身完整保留。

> 📌 这一改**不需要重切数据** —— 截断发生在 `bin_to_latent`（注入时），
> 而 `.npy` 存的是未缩放的逐帧原值。这正是 §4.3 那个设计的第一次兑现。

### 4.2 🔴 捷径学习的风险，以及 `--cam-dropout`

相机增量是动作的**结果**，信息量远大于按键 —— 它几乎直接告诉了模型
「下一帧画面该往哪挪」。**模型完全可能只读这 6 个通道，让 11 个按键通道
彻底不训练。** 那样我们得到的是一个相机轨迹条件模型，不是动作条件世界模型，
而且要到最后做交互推理时才会发现（那时只有按键可用）。

对策是 `inject_abot_action.py --cam-dropout 0.5`：按 `sample_id` 确定性地
把一半 clip 的相机通道整块置零。跨 epoch 固定，等价于把数据集分成
「带相机条件」和「纯按键」两半，强迫按键通路也得干活。

用 `sample_id` 而不是行号播种，是为了换 tier 时同一条 clip 的归属不变。

> **判据**：训练几百步后看存盘权重的逐通道梯度（世界模型文档 §7.0c 那套方法）。
> 如果 11 个按键通道的 `|w|max` 明显低于相机通道，说明捷径仍然存在，
> 应当调大 `--cam-dropout` 或干脆先跑 `--keys-only`。

### 4.3 动作 schema **没有**被 stage 1 锁死

这是这套流程相对 V-Rising 的一个实际改进：`build_abot_clips.py` 把
**逐帧、未缩放**的动作矩阵 `[130, 17]` 存成 `.npy` 放在切片旁边（8.8 KB/条），
`inject_abot_action.py` 只做「截断 + 分箱 + 缩放 + 写进缓存」。

所以想改通道构成、量纲、dropout 比例，**重跑注入脚本几分钟就行**，
不用重跑 stage 1 的 latent 编码。唯一被 stage 2 启动时锁死的是宽度
（`--action_num_buttons`）。

**推论：schema 的选择是低风险可逆决策，不必在开跑前纠结。**
建议第一轮就上 17 通道 + `--cam-dropout 0.5`；若梯度诊断显示捷径，
再退到 `--keys-only`（11 通道）重注入。

### 4.4 模型侧零改动

Stage B 那套已经验过的通路原样可用（世界模型文档 §4）：

- `model_fn_minimax_h3(..., action_cond=None)` 具名形参 → 过缓存白名单
- `enable_action_conditioning(num_buttons)` 延迟建 embedder，零初始化
- 50 层各一个 `Linear(17→5376, bias=False)`，共 `17×5376×50 = 4,569,600` 参数
- 注入点 `narrow`+广播+`cat`，已由 `probe_action_injection.py` 验过避开 keyframe anchor

`Linear` 对连续输入和二值输入没有任何区别，所以**加连续通道不需要改模型代码**。

> ⚠️ `probe_action_injection.py` 应当在 ABot 的缓存上**重跑一遍**。
> 它验的是空间落点，与 `latent_t` 有关（32 → 37），换了数据就该重验。

---

## 5. 产出与规模

### 5.1 目录

```
/nfs/danze/data/abot/
  clips/<prefix>/<sample_id>_w000.mp4     832x480 @24fps 130 帧，~4.6 MB
  clips/<prefix>/<sample_id>_w000.npy     [130, 17] float32，8.8 KB
  abot_manifest.jsonl                     全部成功切片，全局固定顺序
  abot_meta_64.jsonl / _2000 / _8000      manifest 的**前缀**
```

metadata 一行（字段契约见架构文档 §2）：

```json
{"video": "6b/6b489f..._w000.mp4", "input_audio": "6b/6b489f..._w000.mp4",
 "prompt": "The scene is an urban street intersection rendered in a realistic...",
 "action": "6b/6b489f..._w000.npy", "sample_id": "6b489f...", "window": 0,
 "src_start": 1316, "perspective": "third", "narrative": "..."}
```

`input_audio` 指向 mp4 自身 → 加载失败 → `--silent_on_missing_audio` 兜底静音，
与 V-Rising 完全一样（架构文档 §3.1）。**命令行必须带这个开关，否则 crash。**
`action` / `sample_id` / `window` / `src_start` / `narrative` 训练不消费。

### 5.2 🔴 行序纪律：所有 tier 都是前缀

姊妹文档 §7.4b 的教训在这里是**从一开始就按它设计的**，不是事后补救：

- episode 顺序由 `SEED=20260817` 固定打乱，任何机器可复现
- 排布是「先把所有 episode 的第 0 窗排完，再排第 1 窗」——
  于是**加 episode 和加窗口都是纯追加**
- 每个 tier 就是 manifest 的前 N 行，字节级前缀

所以 `abot_meta_64` 的缓存可以直接拷进 `abot_meta_2000-cache/` 复用，
`2000` 的可以拷进 `8000` 的。前提两条（破一条全废）：
**必须单进程 `num_processes=1`**，且**只能往后追加、不能重排前缀**。

### 5.3 窗口放置

每条 episode 切成 `1800 // 162 = 11` 个互不重叠的槽位，
用 `sample_id` 播种打乱槽位顺序，第 w 个窗口取打乱后的第 w 个槽。

两个作用：同一条 episode 的多个窗口天然不重叠（否则相邻窗口是近乎重复的样本，
就是 V-Rising 倒放配对那种冗余陷阱）；**窗口 0 均匀散布在整条 60s 上**，
而不是所有 episode 都从开头切 —— 开头往往镜头刚起步、动作稀疏，全取开头会让
动作分布偏斜。

> 这一条是跑完 64 条冒烟后改的：第一版用 `slot = w % n_slots`，
> 结果所有 w=0 的窗口都落在源帧 0~80。改完 ffmpeg 每片从 0.13s 变 0.30s
> （要多解码一段），值得。

### 5.4 建议的规模档位

**优先 1 窗/episode**（场景多样性最大化），不够再加第 2 窗。

| tier | clips | epoch | 步数 | stage 1 | cache | stage 2 | 用途 |
|---|---:|---:|---:|---:|---:|---:|---|
| **64** | 64 | 5 | 320 | **12.7 min** | **0.86 GB** | ~1.8 h | 冒烟 + 通路验证 |
| **2000** | 2000 | 3 | 6000 | **6.9 h** | **27 GB** | ~33 h | ✅ **首轮正式** |
| 8000 | 8000 | 2 | 16000 | **27.6 h** | **108 GB** | ~89 h | 效果不够时再上 |
| 30969 | 全部 1 窗 | 1 | 31k | ~107 h | 417 GB | ~172 h | 大概率不需要 |

> ⚠️ stage 1/2 的时间按**争抢状态**估（V-Rising 实测惩罚 2.0–2.1×，姊妹文档 §6.7），
> 因为 08-17 当前八张卡 util 全是 100%。独占时约为表中的一半。
> stage 2 单步按 ~20 s/it 估（V-Rising 107 帧争抢时 16.6 s/it，
> 124 帧 video rows +15.6%，attention 二次项让它再涨一些）。**这是外推不是实测。**

**推荐首轮打 2000。** 依据是姊妹文档 §7.1 那笔账：rank-32 LoRA 是 63M 参数，
1000–10000 步收敛，2000×3 = 6000 步正落在舒适区。上 8000 条会把步数推到 16000，
边际收益接近零 —— 这正是 `all`(20699) 那轮被推翻的原因，不要再犯一次。

### 5.4b ✅ 预处理已完成（08-17 实测）

```
8000/8000  1915s  0.24 s/片  成功 8000  失败 0
clips 37 GB      残留 .tmp 0      video/sample_id 去重 8000/8000
前缀嵌套 64 ⊂ 2000 ⊂ 8000 ⊂ manifest  —— 全部字节级通过 ✅
对齐复核 10/10 PASS（124 帧 / 0 重复帧 / 位移扫描 argmin=0）
```

24 workers，32 分钟。**注意缓存体积的大头是 `prompt_embeds` 不是 latent** ——
实测 V-Rising 一条 11.84 MB 里 `prompt_embeds`(848×5120×bf16) 占 8.7 MB，
`input_latents` 只有 2.4 MB。ABot 的 caption 更长（114 词 vs 79 词），但 **text rows 是次线性增长的** ——
实测 964 行（V-Rising 848），不是我按词数比例外推的 1220。
所以每条实测 **13.48 MB**，上表已按实测值填。

**一条脏 caption**：8000 条里有 1 条（行 2280）的 `scene_static` 是字面量 `'!!!'`。
删行会让后面全部错位、缓存全废，所以改为在 `pick_prompt()` 里加退化回退
（`< 30 词就用 narrative`），**只改文本不改行数**，前缀嵌套已复核仍然成立。
修完 prompt 词数 min 从 1 变 49。0 重复 prompt、0 个 `[redacted]`。

### 5.5 磁盘

`/nfs` 当前 **4.0T 可用**（50T 中已用 47T，93%）。2000 档要 28 GB + 9 GB 切片，
8000 档要 112 GB + 37 GB，都够。

> 顺带：`/nfs/danze/model/minimax_h3_vrising/all-cache` 那 **83 GB** 是
> 姊妹文档 §7.4 记的沉没成本，早就确认不需要了，删掉可以直接回收。
> `/`（home）只剩 **49G**，别把 HF/modelscope 缓存落在那。

---

## 6. 怎么跑

```bash
# 0. 自检（纯 CPU，秒级）
python3 /nfs/danze/abot/abot_action.py

# 1. 切片 + metadata。先冒烟
python3 /nfs/danze/abot/build_abot_clips.py --num-clips 64 --workers 16 --tiers 64
python3 /nfs/danze/abot/build_abot_clips.py --verify 8        # 对齐抽查，必跑

# 2. 正式档（前缀保序，会覆盖 manifest 并重新导出 tier）
python3 /nfs/danze/abot/build_abot_clips.py --num-clips 2000 --workers 24 \
        --tiers 64,2000

# 3. stage 1：编码 latent（单进程！否则 data_id 映射失效）
cd /nfs/danze/repo/DiffSynth-Studio-new
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 STAGE=1 \
  bash examples/minimax_h3/model_training/lora/ABot-FL2VA.sh 64

# 4. 注入动作（纯 I/O，几分钟）
python3 /nfs/danze/abot/inject_abot_action.py \
  --meta abot_meta_64.jsonl \
  --cache /nfs/danze/model/minimax_h3_abot/64-cache --dry-run
python3 /nfs/danze/abot/inject_abot_action.py \
  --meta abot_meta_64.jsonl \
  --cache /nfs/danze/model/minimax_h3_abot/64-cache

# 5. 落点探针（换了 latent_t，必须重跑）
python3 /nfs/danze/probe_action_injection.py \
  --cache /nfs/danze/model/minimax_h3_abot/64-cache --n 9999

# 6. stage 2
CUDA_VISIBLE_DEVICES=0 STAGE=2 ACTION_BUTTONS=17 SAVE_STEPS=50 NUM_EPOCHS=5 \
  bash examples/minimax_h3/model_training/lora/ABot-FL2VA.sh 64

# 复用缓存上更大的前缀保序 tier
cp -n /nfs/danze/model/minimax_h3_abot/64-cache/0/*.pth \
      /nfs/danze/model/minimax_h3_abot/2000-cache/0/
```

> 进度条是 `\r` 刷新的，`tail` 直接看是一坨。用 `tr '\r' '\n' < 日志 | tail`。

---

## 7. 待决项与已知缺口

1. **stage 1/2 在 124 帧下的真实速率没测过。** §5.4 的时间全是从 V-Rising
   的 107 帧外推的。冒烟那轮跑完就能拿到真值，**应当先跑 64 条再决定 2000 档的排期**。
2. **捷径学习是否真的发生，未验。** §4.2 给了对策和判据，但要等带动作的
   stage 2 跑出逐通道梯度才知道。这是本方案最大的未知数。
3. **`--cam-dropout 0.5` 这个比例是拍的**，没有依据。若诊断显示按键通路弱，
   往上调；若相机条件学得不好，往下调。
4. **第三人称视角与「世界模型」的匹配度存疑。** 255/256 是 third person，
   画面里有一个玩家角色，动作同时驱动角色和相机。V-Rising 是等距视角，
   ReactiveGWM 是格斗游戏。三者都不是第一人称。这不阻碍训练，但
   「按键 → 画面变化」的映射里混进了角色动画这一层，比纯相机控制更难学。
5. **prompt 变长的代价没量化。** `scene_static` 中位 114 词 vs V-Rising 的 79 词，
   text rows 会从实测的 906 涨到大约 1300。相对 video 的 14430 行只占 9%，
   预计无影响，但冒烟那轮应当读一下真实 `seq_len` 确认。
6. **Q/E/Space 三个恒零通道**已在 **2225 条切片**上确认精确为零（原先只有 296 条抽样），
   全集 30969 仍未扫。留着不管即可 —— 代价是 3×5376×50 个拿不到梯度的参数，
   而它们正好充当 §4.2 梯度诊断的**阴性对照**（应当精确为 0，非零就说明串列了）。
7. **`num_frames` 是否该上 141/175。** 世界模型的时序上下文越长越好，
   而这份数据不再有时长约束 —— 唯一的代价是算力。Stage C（history-conditioned
   rollout）时应当重新算这笔账，那时时间连续性比省算力重要。

---

## 8. 文件索引

| 路径 | 说明 |
|---|---|
| `/nfs/danze/abot/abot_action.py` | 动作 schema + COLMAP 反推 + 非均匀分箱（带 self-test） |
| `/nfs/danze/abot/build_abot_clips.py` | 切片 + metadata + 动作 npy；`--verify` 做对齐抽查 |
| `/nfs/danze/abot/inject_abot_action.py` | 缓存事后注入（`--cam-dropout` / `--keys-only`） |
| `.../lora/ABot-FL2VA.sh` | 训练入口（STAGE / ACTION_BUTTONS / NUM_EPOCHS 可配） |
| `/nfs/danze/data/abot/abot_manifest.jsonl` | 全局固定顺序的切片清单 |
| `/nfs/danze/probe_action_injection.py` | 注入落点探针（复用，换 latent_t 后必跑） |
| `<源>/abot-world-explorer-colmap-skill/SKILL.md` | 官方处理规范（tar 白名单、COLMAP 约定、不许臆测的清单） |

---

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 08-18 | **冒烟 stage 1 完成并核对**：64 条 / 12分41秒 / 11.9 s/clip / 0.86 GB（13.48 MB 每条）。四项判据全过 —— `latent_t=37`（124 帧未被降级）、真实 VAE 输出与 `frame_spans` 推导对撞成立、`seq_len=16640`（text 964 + anchor 780 + audio 414 + video 14430）。**修正**：text rows 是次线性增长的（114 词→964 行，非按词比例的 1220），所以各档缓存估算改回 27/108 GB，我此前上调到 31/126 是错的。动作注入 64/64、落点探针 **64/64 PASS**。`cam-dropout` 在 n=64 上偏到 62.5% 但 2000/8000 档均为 50.5%，属小样本噪声，代码未动 |
| 08-17 最后 | **8000 条预处理完成**（32 min / 0.24 s/片 / 零失败 / 37 GB），前缀嵌套 64⊂2000⊂8000⊂manifest 字节级通过，对齐复核 10/10。**三处修正**：episode 帧数**不都是 1800**（见到 2348），`src_start` 上限随之变化；缓存体积大头是 `prompt_embeds` 不是 latent，各档估算上调约 12%；发现 1 条 `scene_static='!!!'` 的脏 caption，加 `pick_prompt()` 退化回退（只改文本不改行数，前缀嵌套已复核） |
| 08-17 晚些 | **在 2225 条上复核动作分布，发现旋转也有重尾**（`d_yaw` max 21.15 而 p99 仅 1.18，64 条样本上看不出来），新增 `ROT_CLIP=3.0`；依据是分位数拐点 + 「极值 token 只有 15% 按着转向键」。Q/E/Space 恒零的证据从 296 条抽样加强到 2225 条切片。**不需重切数据**（§4.3 的设计兑现） |
| 08-17 | 初版。§1 源数据实测（30969 条 / 60s / 1080p30 / 无音轨 / 2.5TB）、§2 动作数据（**`delta_*` 全零**、COLMAP 满覆盖但尺度差 33×、8 键因果核对）、§3 切片方案（124 帧、30→24 丢帧规则、**0 重复帧且对齐 argmin=0 实测**、prompt 选 `scene_static` 的理由）、§4 17 通道注入与捷径学习风险、§5 前缀保序与规模档位、§6 跑法、§7 七条待决项 |
