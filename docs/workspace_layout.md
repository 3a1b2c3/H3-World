# 工作机布局与环境（机器相关）

> 本文记录这套代码在开发机上的落地方式：权重放哪、缓存钉在哪、怎么起训。
> 项目本身是什么、当前方案、怎么复现，见仓库根目录的 `README.md`。
> 注意本文写于「动作张量注入」阶段，§5 的起训命令与 §6 的 blocker 已被
> `docs/pipeline_text_injection.md` 取代，保留是为了环境与路径部分仍然有效。

本目录 = COS 上的 `minimax_h3_bundle.tar.gz`（代码/文档/metadata）+ `abot_clips_8000.tar.gz`
（38.5 GB 视频切片）在本机的落地。**原 bundle 的路径全是 `/nfs/danze/...`，本机没有 `/nfs`**，
所以 `code/` 下的路径常量已改写为本目录（改法见 §3）。

> bundle 原始 README 在 `docs/BUNDLE_README.md` —— **设计论证、判据、踩坑清单都在那里**，
> 本文只讲「在这台机器上东西放在哪、怎么起」。
>
> **这条线做到哪了** —— 见 `docs/PROGRESS.md`（时间线、当前训练状态、下一步判据）。
>
> **动作注入为什么没效果、该怎么改** —— 见 `docs/action_injection_design.md`。
> 那里有实测诊断（注入量级只有 hidden 的 0.5‰、梯度互相抵消）、H3 三条架构约束，
> 以及 LingbotWorld / Matrix-Game / Incantation 三个方案的适配性评估。

---

## 1. 目录布局

```
minimax_finetune/
├── README.md                    项目说明（GitHub 门面）
├── code/                        ← 已改好本机路径，跑这份
│   ├── abot/                    ABot 线：动作 schema / 切片 / 注入
│   ├── vrising/                 V-Rising 线（路径未改，仍是 /nfs/...；probe 通用）
│   ├── scripts/ABot-FL2VA.sh    训练入口（两阶段）
│   └── abot/viz_action.py       动作可视化（本机新增，见 §7）
├── data/
│   ├── clips/<prefix>/<sid>_w000.{mp4,npy}   8000 条切片 + 逐帧动作，37 GB
│   ├── abot_meta_{64,2000,8000}.jsonl        训练 metadata（互为字节前缀）
│   └── abot_manifest.jsonl                   = abot_meta_8000.jsonl（md5 同）
├── docs/
│   ├── PROGRESS.md                  ★ 进展记录（现在在哪、怎么走到这、下一步）
│   ├── action_injection_design.md   ★ 动作注入方案分析（失效诊断 + H3 适配性 + 推荐方案）
│   ├── injection_options.html       方案选型可视版（框图 + 判定 + 计划）
│   └── ...                          bundle 的 6 篇设计文档 + 原 README/MANIFEST
├── bundle/minimax_h3_bundle/    原始 bundle 解压件，只读留档
├── DiffSynth-Studio-h3/         已 checkout 到 base commit 并打好补丁的框架
│   └── models/MiniMax/MiniMax-H3/FL2VA/   权重 134 GB
├── output/
│   ├── viz/                     动作可视化渲染结果（见 §7）
│   └── minimax_h3_abot/<N>[-cache]        训练产出：latent 缓存与 LoRA
└── logs/                        下载/解压日志
```

## 2. 已验证

| 项 | 结果 |
|---|---|
| 两个包下载 | 字节数与 COS 端一致（36 GB @ 677 MB/s，54 s） |
| tier 前缀性质 | 64 ⊂ 2000 ⊂ 8000 **逐字节**成立 |
| metadata ↔ 文件 | 8000 行全部命中，缺 video 0 / 缺 action 0 |
| action 张量 | `(130, 17) float32` — 与 README 期望一致 |
| `abot_action.py` 自检 | 全部通过：`latent_t=37`、有效帧率 24.156、token0=0.250/token1=1.000 |
| 补丁应用 | `git apply --check` 干净通过，98 行 / 4 文件 |
| 补丁接口生效 | `model_fn_minimax_h3` 有**具名** `action_cond`、`DiT.enable_action_conditioning` 存在 |
| 环境 | `minimax_h3` 环境 import diffsynth 指向本 checkout，torch 2.10.0+cu128，8×H200 可见 |
| stage 1 冒烟 | 64/64，12.9 s/条；`latent_t=37`、audio `(2,32,207)`、anchor `(780,96)`、`seq_len=16640` 全部命中 |
| 动作注入 | 64 条，相机置零 40；按键覆盖与相机 \|max\| **与 bundle README 记录逐项一致** |
| 落点探针 | **64/64 通过** —— narrow 实现与 index_add 逐位一致、bias 只落 video 行、anchor 零污染、无错位 |
| stage 2 通路 | `action_embedders` 4,569,600 参数建立成功，`action_train_only` 下可训练总数与之相等，训练已迭代 |
| stage 2 显存 | **79 GB**（未用 fp8、未用 offload），22 s/step |
| 判据一 通道对位 | ✅ Q/E/Space |w|max **精确为 0**，其余 14 通道全部有梯度 —— 无串列 |
| 判据二 捷径学习 | 按键/相机 |w|max = **1.19x**，量级相当 —— 50 步时按键通路确实在训 |

## 3. 相对 bundle 改了什么

只改路径，逻辑零改动。全部可用环境变量覆盖回去：

| 文件 | 常量 | 现在的默认值 | 覆盖变量 |
|---|---|---|---|
| `code/abot/build_abot_clips.py` | `OUT_ROOT` | `<本目录>/data` | `ABOT_OUT_ROOT` |
| 同上 | `SRC_ROOT` | **仍是 `/nfs/yixinyang/...`（本机无）** | `ABOT_SRC_ROOT` |
| 同上 | `sys.path` 里的 repo | `<本目录>/DiffSynth-Studio-h3` | `DIFFSYNTH_ROOT` |
| `code/abot/inject_abot_action.py` | `OUT_ROOT` | `<本目录>/data` | `ABOT_OUT_ROOT` |
| `code/scripts/ABot-FL2VA.sh` | `DATA_BASE` / `META` / `OUT` | `data/clips`、`data/abot_meta_N.jsonl`、`output/minimax_h3_abot/N` | 同名变量 |
| 同上 | `cd` 到 repo 根 | `${DIFFSYNTH_ROOT:-<脚本上溯四级>}` | `DIFFSYNTH_ROOT` |

`SRC_ROOT` 只在**重建切片**时才用得到，而切片已经现成了，所以留原值不影响训练。

`code/vrising/` 的路径**没改**（那条线的数据不在本机）。但 `probe_action_injection.py`
是两条线通用的，用 `--cache` 传目录即可，不受影响。

## 4. 框架 / 权重 / 环境（均已就位）

### 4.1 DiffSynth checkout + 补丁 ✅

`DiffSynth-Studio-h3/` 是从 `modelscope/DiffSynth-Studio` 新 clone 的一份，已 checkout 到
补丁的 base commit `300e3e4`（"Minimax h3 pruned nf4 #1584"）并打好
`code/diffsynth_h3_action.patch`。

> 你原来那份 `/opt/dlami/nvme/danze/DiffSynth-Studio` 里找不到 `300e3e4`，纯粹是它没
> fetch 到那个 commit —— 新 clone 里有。**那份仓库没有被动过。**

补丁状态：`git status` 显示 4 个文件 modified，`git diff --stat` 98 行。工作树停在
detached HEAD，要看改了什么直接 `git diff`。

### 4.2 模型权重 ✅ 134 GB

只下了 `FL2VA/` 子目录 —— 整个 repo 是 464 GB，另一半是 `Ref2VA` / `transformer_ref`，
这条线用不上：

```
FL2VA/text_encoder   62.1 GB      FL2VA/video_vae   9.7 GB
FL2VA/transformer    61.7 GB      FL2VA/audio_vae   0.6 GB
```

落点 `DiffSynth-Studio-h3/models/MiniMax/MiniMax-H3/FL2VA/`，即 DiffSynth 默认布局，
脚本里的 `--model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/..."` 不用改。
`HF_HOME` 设在 `/opt/dlami/nvme/danze/.hf_home`（不在 home 盘上）。

要补下 `Ref2VA` 时：`hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" --local-dir <同一目录>`。

### 4.3 conda 环境 `minimax_h3` ✅

从 `mg3` clone 出来（省掉 torch 重装），再 `pip install -e ".[audio]"`：

```bash
conda activate minimax_h3                              # 推荐
/opt/dlami/nvme/danze/envs/minimax_h3/bin/python       # 或直接用解释器
```

**环境建在 nvme 上**（`/opt/dlami/nvme/danze/envs/`），不在根盘 —— 根盘 `/` 只有 100G，
而 `/opt/dlami/nvme` 是 27T。两个都是 instance store（`lv_ephemeral`），持久性一样，
所以放大盘没有额外代价。`~/.condarc` 里 `envs_dirs` 第一项已指向那里，
所以 `conda activate minimax_h3` 按名字就能命中，不用写全路径；
以后 `conda create -n xxx` 也默认建在 nvme 上。

torch 2.10.0+cu128 / Python 3.10.20，`audio` extras（av、torchaudio、torchcodec、librosa）
已装 —— H3 有 audio VAE，缺了 stage 1 会挂。`diffsynth` 是 editable 装的，指向
`DiffSynth-Studio-h3/`，所以**改补丁立刻生效**，不用重装。

> 仍然要 `export PYTHONPATH`（README §2.2）：`accelerate launch` 时 `sys.path[0]` 是脚本
> 目录，editable 安装救不了这一条。`ABot-FL2VA.sh` 里已经带了。

## 4.4 缓存全部钉在 nvme（三层，不依赖记性）

根盘 `/` 只有 100G。而 HF、pip、torch hub、triton、torchinductor、modelscope、
matplotlib **默认全往 `~/.cache` 和 `/tmp` 写**（实测 9 个路径无一例外），
其中 torchinductor 在加载 62G DiT 时可能写出几百 MB。

早先的做法是手动 `source env.sh` —— 靠记性，忘一次就前功尽弃。现在改成三层：

| 层 | 位置 | 覆盖 |
|---|---|---|
| ① conda 钩子 | `$CONDA_PREFIX/etc/conda/activate.d/zz_cache_to_nvme.sh` | `conda activate minimax_h3` |
| ② sitecustomize | `<env>/lib/python3.10/site-packages/sitecustomize.py` | **直接调解释器全路径**（钩子管不到的场景） |
| ③ conda pkgs_dirs | `~/.condarc` | `conda install` 的包缓存 |

②是关键的一层：`envs/minimax_h3/bin/python xxx.py` 这种用法绕过 activate 钩子，
而 Python 启动时一定会 import `sitecustomize`（在任何用户代码之前），
所以对 `huggingface_hub` 这类**在 import 时就把路径固化成模块常量**的库刚好来得及。

两层都用 `setdefault` / 保存旧值，**不会覆盖外部显式指定的值**，deactivate 时干净还原。

实测（`env -i` 清空所有变量、不激活环境、直接调解释器）：

```
HuggingFace / torch hub / triton / torchinductor / pip / modelscope / matplotlib
→ 全部落在 /opt/dlami/nvme/danze/cache/
```

`env.sh` 保留着，但现在只是给非 conda 场景兜底 —— 正常用 `conda activate minimax_h3` 就够。

### 巡检

理论覆盖不等于实际覆盖，总有工具用硬编码路径。定期跑：

```bash
bash code/check_rootfs.sh                    # 默认看最近 1 小时
bash code/check_rootfs.sh "2026-08-20 04:30" # 或指定起点
```

它列出根盘上被写过的位置并在超 85% 时告警。已知的两个常驻噪声：
`/root/.local/share/CodeBuddyExtension`（IDE 扩展，不是这条线的）、
`/root/.conda/environments.txt`（conda 的环境注册表，几十字节，避不开）。

## 5. 起训

```bash
cd /opt/dlami/nvme/danze/minimax_finetune
export DIFFSYNTH_ROOT=$PWD/DiffSynth-Studio-h3
export PATH=/opt/dlami/nvme/danze/envs/minimax_h3/bin:$PATH

# stage 1：缓存 latent（64 条冒烟约 13 分钟）
CUDA_VISIBLE_DEVICES=0 STAGE=1 bash code/scripts/ABot-FL2VA.sh 64

# 注入动作（纯 CPU，分钟级）；先 --dry-run
python3 code/abot/inject_abot_action.py --meta abot_meta_64.jsonl \
        --cache output/minimax_h3_abot/64-cache --dry-run
python3 code/abot/inject_abot_action.py --meta abot_meta_64.jsonl \
        --cache output/minimax_h3_abot/64-cache

# 落点探针 —— 改了序列布局或 latent_t 后必跑，期望 64/64
python3 code/vrising/probe_action_injection.py \
        --cache output/minimax_h3_abot/64-cache --n 9999

# stage 2
CUDA_VISIBLE_DEVICES=0 STAGE=2 ACTION_BUTTONS=17 ACTION_TRAIN_ONLY=1 \
  SAVE_STEPS=50 NUM_EPOCHS=5 bash code/scripts/ABot-FL2VA.sh 64
```

## 6. 两个原 blocker 在这台机器上的状态

**§6.2 显存阻塞 —— 已实测解除。** stage 2 冒烟在本机跑起来了，实测占 **79 GB**，
与原 README 估的 "~79 GB" 完全吻合，**没用 `--fp8_models dit`、没用任何 offload**。
卡是 143 GB × 8，占卡进程（`../occupy_gpu.py`，每卡 60 GB）不动也够跑；停掉它更宽裕。

这一点很重要：原方案被迫用 fp8 是显存所迫，而 fp8 **会改变权重数值**、不适合评估效果
（只适合通路诊断）。在这台机器上这个妥协不需要了，冒烟结果可以直接用来判断效果。

**§6.1 推理侧代码缺口 —— 诊断已更正，修复面比原描述小得多。**

原 README 说 `code/UNAPPLIED_inference_action_cond.rej` 是"静默打失败的 hunk"。
实测**不是打不上**：

```bash
git apply -p0 code/UNAPPLIED_inference_action_cond.rej    # ✅ 直接通过（自动处理 offset -12）
```

它被判为 UNAPPLIED 只是 **`-p` 层级用错**了 —— 主补丁 `diffsynth_h3_action.patch` 是
`git diff` 生成的、带 `a/` `b/` 前缀，默认 `-p1`；而 `.rej` 是 `patch(1)` 生成的、
不带前缀，必须 `-p0`。用 `-p1` 去打它，路径被剥成 `pipelines/...`，报的是
"No such file or directory"，看着像冲突，其实是找不到文件。

**但光打上这个 hunk 不够 —— 它只有一半。** hunk 只往 `__call__` 签名里加了形参，
而 `model_fn` 是靠 `**inputs_shared` 取参的（`base_pipeline.py:327`
`model_fn(**inputs_posi, **inputs_shared, **inputs_others)`）。不把 key 放进那个 dict，
签名上的 `action_cond` 就是个**死参数**。

完整修复 = hunk + 一行：

```python
inputs_shared = {
    ...,
    "action_cond": action_cond,      # ← .rej 缺的正是这一半
}
```

已在副本上验证四段链路全通（`__call__` → `inputs_shared` → `cfg_guided_model_fn`
→ `model_fn` 具名形参 → `dit.forward`）。units 不会剪掉多余 key（只 `.get` + `update`），
所以加这个 key 是安全的。**本仓库尚未应用**，等推理验证一起做。

### ⚠️ cam-dropout 给 62% 的样本贴了错误标签

`inject_abot_action.py --cam-dropout 0.5` 的本意是防捷径学习：把一半样本的相机通道
整块置零，逼模型去读按键。但它有两个实现层面的问题：

**① 置零 ≠ 缺失。** 零向量在数值上就是"相机静止"，模型无法区分"这条信息没给你"和
"相机确实没动"。64 条冒烟里被置零的 40 条，**没有一条**接近静止：

```
被置零 40 条，真实相机运动 |max|：中位 1.830   p90 3.620   最大 4.000
接近静止（|max| < 0.1）的：0 / 40
```

也就是 62.5% 的样本拿到的是一个和画面矛盾的条件：输入说"相机没动"，
而画面里相机在明显运动。这不是在"隐藏信息"，是在"给错信息"。

**② 是静态划分，不是 dropout。** 它按 `sample_id` 确定性地决定，且**写进缓存**，
所以同一条 clip 在所有 epoch 里状态固定。标准 dropout 是每次前向随机，
让同一个样本在有/无该输入两种情况下都能预测；静态划分做不到这件事，
只是把数据集切成了两半。

**建议**：加第 18 维做 `cam_valid` 标志位（1=相机通道有效，0=已屏蔽），
让"缺失"和"零"可区分；或者把置零改成一个可学习的 `unknown` 嵌入。
要在注入阶段改，`--action_num_buttons` 跟着改成 18。

### 顺带发现的两个问题（都在推理路径上）

**① device 不匹配（会直接崩）。** `minimax_h3_dit.py:416`：

```python
emb = self.action_embedders[i](action_cond.to(hidden.dtype))   # 只转 dtype，没转 device
```

训练时 `transfer_data_to_device` 会递归把张量搬上 GPU，所以不触发；推理时用户手传的
张量通常在 CPU，实测直接 `RuntimeError: Expected all tensors to be on the same device`。
修法：`action_cond.to(device=hidden.device, dtype=hidden.dtype)`。

**② 动作不参与 CFG。** `action_cond` 放在 `inputs_shared` 里，正负两侧前向共用同一份，
所以 `cfg_scale` 放大不了动作控制力。想要"动作 CFG"（负侧传零动作）得把它拆到
`inputs_posi` / `inputs_nega`。这不是 bug，是个设计选择 —— 但如果冒烟发现控制力弱，
这是唯一能加的旋钮，值得提前知道。

---

## 7. 动作可视化 `code/abot/viz_action.py`

把 `clips/<sid>_w000.npy` 的逐帧动作渲染成 HUD，和切片视频并排输出成 mp4。

```bash
export PATH=/opt/dlami/nvme/danze/envs/minimax_h3/bin:$PATH
python3 code/abot/viz_action.py --n 6 --pick active     # 挑动作最丰富的 6 条
python3 code/abot/viz_action.py --sample <sample_id>    # 指定样本
```

约 **2 秒渲一条**（130 帧），输出到 `output/viz/<sample_id>_action.mp4`，1232×704 @24fps。
已渲好 6 条在 `output/viz/`。展示页（内嵌 3 条 + 结论）：`output/viz/action_readout.html`。

### 为什么值得做

`build_abot_clips.py --verify` 的位移扫描 argmin 判据只回答「有没有错一帧」，
回答不了「**动作语义对不对**」—— COLMAP 反推出来的平移方向是否真和按键一致、
`d_yaw` 的正负号是否真对应画面转向。那是会静默错的，训练也照样收敛。
这个脚本让它变成肉眼一眼可判的事。

### 转向符号：6/6 实证

把 J、L 的按下帧数相减，和整条 clip 的累计 yaw 比符号（`--pick active` 的 6 条）：

| 样本 | J 帧 | L 帧 | J − L | 累计 yaw | 判定 |
|---|--:|--:|--:|--:|---|
| `504032a4` | 95 | 24 | +71 | +22.5° | 同号 |
| `6ed0158e` | 88 | 13 | +75 | +36.1° | 同号 |
| `7efa2dd0` | 34 | 76 | −42 | −10.2° | 同号 |
| `4b3d12db` | 0 | 130 | −130 | −67.1° | 同号 |
| `0af6ec90` | 0 | 130 | −130 | −136.5° | 同号 |
| `3d16a622` | 0 | 0 | 0 | −4.7° | **近零 · 阴性对照** |

**J = 左转 → yaw 正，L = 右转 → yaw 负**，6 条无一例外；没按视角键的那条落在零附近。
这条此前只有推导没有实证。

### ⚠️ 但平移通道不能这么读

**不要拿 W/A/S/D 去对累计平移量。** 平移记在**前一帧相机系**里，相机一直在转，
累加出来的 x/z 没有全局意义 —— `0af6ec90` 按了 88 帧 D（右移）却累计出 x −126.65，
看着矛盾，其实只是相机在这 130 帧里转了 136 度。

更根本的是 COLMAP 反推的是**相机**位姿，第三人称下相机 ≠ 角色输入：
`3d16a622` 按了 47 帧 S（后退）而 `d_z_fwd` 累计为正，是相机跟随的结果，不是标注错了。

### 布局

```
┌──────────────────────┬──────────┐
│                      │ 键盘     │  右栏 400px：WASD/IJKL 九宫格 + QE/Space
│   视频 832×480 原样  │ 罗盘     │  罗盘 = 累积 yaw 朝向 + pitch 竖条
│                      │ 俯视轨迹 │  轨迹 = x-z 平面累积路径（归一化，只有形状有意义）
├──────────────────────┴──────────┤
│ 17 通道 × 130 帧 时间线 + 游标  │  下条 224px：整条 clip 的动作结构
└─────────────────────────────────┘
```

几个刻意的取舍：

- **不遮挡画面** —— HUD 全在黑边里。要判断「动作和画面一致吗」，画面就不能被盖住。
- **底部时间线是全局视图** —— 只看当前帧看不出结构（哪段在走、哪段在转）。
  按键行画细、相机行画粗：按键是二值的几像素就够读，相机增量要看幅度。
- **罗盘用累积量不是当前帧增量** —— 单帧 ~1 度画出来是看不见的抖动，
  累积朝向才和画面里看到的转向对得上。
- **Q/E/Space 恒零也画出来** —— 它们是天然的阴性对照（§6.3 判据里要求梯度精确为 0），
  画出来才能看见「它一直不亮」这件事。

### 捷径学习的直观证据

`0af6ec90` 那条里 `d_pitch` 整行偏蓝（持续为负），但 I/K（抬头/低头）**一次都没按**。
说明 pitch 变化来自地形起伏和相机跟随，不是玩家输入 —— 这正是
`docs/BUNDLE_README.md` §6.3 担心的**捷径学习**的直观证据：相机通道携带的信息
远多于按键通道，模型可能只读相机就把 loss 降下去。`--cam-dropout 0.5` 是针对它的预防，
但比例是拍的，值得在冒烟 stage 2 的逐通道梯度里重点看。

### 字体的坑

系统里没有一个字体能把中英文都画好：DejaVu 缺 CJK（方框），而 Droid Sans Fallback
是 fallback-only 字体，**小字号下拉丁和数字也是方框**。PIL 又不做字体回退。
所以脚本里有个 `text2()` 按 CJK 边界切段、两个字体分别画。改文案时走 `text2`，
别直接 `d.text`，否则中文或数字会变方框。
