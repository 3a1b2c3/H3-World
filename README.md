# minimax_finetune

给 **MiniMax-H3**（音视频联合的视频扩散 Transformer）加上**动作可控**：给定首帧和一段
按键序列，生成出符合指令的视频。最终目标是流式续写 + 动作可控的世界模型。

数据是第三人称开放世界游戏录像（ABot-World-Explorer），8000 条 124 帧 @24fps / 832×480
的切片，配套逐帧按键与 COLMAP 反推的相机位姿。

---

## 现在的方案：逐 latent 动作文本注入

条件不走特征空间，走 **H3 预训练过的文本通路**：每个 latent 一条结构化文本，
由 **9 位按键的纯函数**生成，训练与推理走完全同一条路径。

```
头部（原样保留）  <Picture 1>: [首帧视觉 392 行] + 场景描述
逐 latent（37 条） the man <怎么动>, camera <相机视角怎么动>
```

三个要点：

- **条件通路零新增可训练参数** —— 「按键 → 文本」是手写规则表，「文本 → 嵌入」是冻结的
  文本编码器。前两次失败的病灶（动作层权重梯度互相抵消）在这条路上不存在。
- **硬绑定靠掩码，不靠位置** —— 自注意力对 token 置换等变，光把 37 条标注塞进序列，
  模型无从知道哪条对哪帧。`score_mod` 让标注行 k 与第 k 帧以外的 video 行互相不可见。
- **第 9 位 `F`(fast) 从 COLMAP 实测速率合成** —— 方向能从 IJKL 读出（命中率 0.85–0.97），
  速度读不出来（按 J 的步 slowly/sharply 是 0.66/0.34，接近抛硬币）。这一位是整个方案里
  唯一真正超出原始 8 bit 的信息。

完整设计与推导：[`docs/action_text_injection_plan.html`](docs/action_text_injection_plan.html) ·
管线状态与实测：[`docs/pipeline_text_injection.md`](docs/pipeline_text_injection.md) ·
文档索引：[`docs/README.md`](docs/README.md)

### 走到这一步之前失败的两次

| 方案 | 注入点 | 调制量级 | ΔW·W 余弦 | 结论 |
|---|---|---:|---:|---|
| 加性 bias | block 入口 | 0.107% @2952 步 | +0.064 | 失效 |
| FiLM | AdaLN 之后 | 0.33% @3000 步 | +0.047 | 失效 |

FiLM 修好了加性 bias 的表达能力缺陷、量级涨了 3 倍，但方向余弦反而更低。
换机制没用，说明问题在**信号本身**而不是注入形式——这是转向文本注入的直接原因。
20 种注入方式的横向评估见 [`docs/injection_options.html`](docs/injection_options.html)。

### 零训练探针的结论

基座模型（不训练）同首帧同 seed，只改 prompt 尾句，用相位相关估全局水平位移
（符号已标定：`cum_dx` 与 `Σd_yaw` 相关系数 **r = −0.936**，14/14 反号）：

| 样本 | left − right | 像素 MAE | 判定 |
|---|---:|---:|---|
| `a3ad9c24` | −116 | 43.26 | 强 |
| `33b38f0f` | −4 | 13.56 | 空 |
| `2c75323e` | −116 | 50.22 | 强 |

**2/3 强阳性** —— H3 的文本通道本来就握着相机运动的控制权。且**动作零参考负 prompt**
把弱方向放大约 9 倍（用空负 prompt 的 CFG 反而更差）。

---

## 仓库里有什么

```
code/
  abot/                     当前这条线
    abot_action.py            动作 schema：COLMAP 解析、分箱到 latent 时间轴
    action_script.py          规则表：9 位按键 -> 结构化标注（纯函数）
    build_abot_clips.py       从源视频切片 + 逐帧动作
    split_abot_metadata.py    训练/测试划分
    inject_abot_text.py       ★ 数据构建：重写 prompt_embeds + packed
    verify_text_cache.py      ★ 缓存完整性校验（六条判据，支持分片）
    infer_abot.py             推理（文本注入 / 旧的动作张量两种模式）
    probe_action_mask.py      ★ 掩码落点探针（改 DiT 后必跑）
    probe_text_channel.py     零训练探针：文本通道控不控运动
    analyze_probe.py          探针判据：相位相关估全局平移
    build_action_viz.py       标注核对台（视频 + 逐 latent 标注同步高亮）
    build_compare_page.py     GT / Generated 横版对比页
    merge_runs.py             多卡各出一条时，合并成一个结果目录
    viz_action.py             动作 HUD 渲染
    check_page_js.py          可视化页的运行时检查（假 DOM 跑 script）
    inject_abot_action.py     旧路径：动作张量注入（ACTION_MODE=cond，仅供对照）
    compare_action_ckpt.py    旧判据：动作层权重差分（对新方案失效）
    check_action_grad.py      旧探针：动作层梯度是否流动
  scripts/ABot-FL2VA.sh     训练入口（两阶段：缓存 latent -> 训 LoRA）
  vrising/                  前一阶段（V-Rising 数据）的脚本，保留备查
  diffsynth_h3_action.patch DiffSynth 框架的全部改动（5 文件 623 行）
  diffsynth_base_commit.txt 补丁对应的 base commit
docs/                       见 docs/README.md
env.sh                      环境变量
```

**不在仓库里**（见 `.gitignore`）：框架与 134 GB 权重、38.5 GB 视频切片、
68 GB latent 缓存与 checkpoint、运行时缓存与日志。

---

## 复现

```bash
# 1. 框架：checkout 到 base commit 再打补丁
git clone <DiffSynth-Studio> DiffSynth-Studio-h3
git -C DiffSynth-Studio-h3 checkout $(cat code/diffsynth_base_commit.txt)
git -C DiffSynth-Studio-h3 apply ../code/diffsynth_h3_action.patch

# 2. 权重：MiniMax-H3 FL2VA 放到 DiffSynth-Studio-h3/models/MiniMax/MiniMax-H3/
#    环境与缓存落盘规则见 docs/workspace_layout.md

# 3. 数据：切片 + metadata
python3 code/abot/build_abot_clips.py --num-clips 8000
python3 code/abot/split_abot_metadata.py

# 4. stage 1：缓存 latent（text encoder + video VAE + audio VAE）
NUM_PROCESSES=8 STAGE=1 bash code/scripts/ABot-FL2VA.sh 7872

# 5. 数据构建：把逐 latent 文本条件写进缓存
#    视频 latent 不动，换标注方案只需重跑这一步
python3 code/abot/inject_abot_text.py \
    --meta data/abot_meta_train_7872.jsonl \
    --cache output/minimax_h3_abot/7872-cache --dry-run   # 先试算
python3 code/abot/inject_abot_text.py --meta ... --cache ...   # 再写入
python3 code/abot/verify_text_cache.py --cache output/minimax_h3_abot/7872-cache

# 6. stage 2：训 LoRA
NUM_PROCESSES=8 STAGE=2 OUT=output/minimax_h3_abot/7872_text \
    bash code/scripts/ABot-FL2VA.sh 7872

# 7. 推理（cfg_scale 是动作服从度的放大器，负 prompt 自动用动作零参考）
python3 code/abot/infer_abot.py --checkpoint <ckpt> --cfg-scale 5.0 --num-samples 8
```

改动 DiT 之后**必须**先跑掩码探针，四条判据全过才继续：

```bash
python3 code/abot/probe_action_mask.py
```

---

## 实测数字

| | |
|---|---|
| 标注词表 | 60 种 = 9 移动从句 × 16 相机从句的实际共现，单条 clip 平均 5.0 种 |
| 数据构建 | 7872 条，8 卡分片约 10 分钟；全量校验 7872/7872 通过 |
| 序列长度 | `seq_len` 15744–15872 → 16064–16448（均值 16237，+5.4%） |
| 镜像偏移 | 标注与自己那帧的 t 偏移恒为 **−201**，7872 条全同 |
| 掩码构建 | 16k 序列上 12.3 ms/forward，50 层共用，占训练步 0.12% |
| 掩码代价 | 注意力从 flash 切到 FlexAttention，**慢 23%**（2.41 → 2.97 s/it） |

硬件：8 × H200（143.7 GB）。

---

## 状态

- [x] 方案设计与横向评估
- [x] 零训练探针（文本通道确实控运动）
- [x] 规则表 + 9 位按键表示
- [x] 掩码 + 版面改造，落点探针 4/4
- [x] 数据构建 + 全量校验
- [x] 推理侧接线 + 端到端冒烟（cfg=1 / cfg=5 均通过）
- [ ] 正式训练
- [ ] 新判据工具（只改第 k 条标注，看输出哪些帧变了）
- [ ] 流式续写（块因果掩码 + KV cache + 少步蒸馏）
