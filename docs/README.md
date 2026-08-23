# 文档索引

按「当前有效 / 阶段性记录 / 背景资料」分三层。**只想知道现在怎么回事，看前两篇就够。**

## 当前有效

| 文档 | 内容 |
|---|---|
| [`action_injection_arch.html`](action_injection_arch.html) | **架构图**。按键怎么变成逐 latent 文本、序列版面与硬绑定掩码、与两次失败方案的结构差异 |
| [`longer_video.md`](longer_video.md) | **能不能生成更长的视频**。当前 5.2s 是怎么来的、架构支不支持、三条路各自的代价与建议顺序 |
| [`journey.md`](journey.md) | **过程记录**。每次方案怎么定的、撞上什么问题、怎么定位、怎么修 —— 包括两次失败的诊断、几个静默 bug、判据本身的演进 |
| [`pipeline_text_injection.md`](pipeline_text_injection.md) | **管线状态**。方案要点、版面与绑定、代码改动清单、全部验证结果、怎么跑、未完成项 |
| [`action_text_injection_plan.html`](action_text_injection_plan.html) | **方案定稿**。为什么是这个方案、信号设计、注入设计、H3 家底对照、流式约束、风险 |
| [`action_prompt_viz.html`](action_prompt_viz.html) | **标注核对台**。8 条片段，播放时按键与标注同步高亮；页尾是 9 位按键 → 文本的映射表 |
| [`injection_options.html`](injection_options.html) | 20 种动作注入方式的横向评估，按在 H3 上的可行度排序 |
| [`workspace_layout.md`](workspace_layout.md) | 开发机上的路径、权重、conda 环境、缓存落盘规则（§5 起训命令已过时） |

> 三个 HTML 都是自包含的：视频以 data URI 内嵌，clone 之后用浏览器直接打开即可，
> 不需要起服务、不依赖外部资源。`action_prompt_viz.html` 由
> `code/abot/build_action_viz.py` 生成，换数据或改规则表后可重建。

## 阶段性记录（已被取代，保留是为了可追溯）

| 文档 | 说明 |
|---|---|
| [`PROGRESS.md`](PROGRESS.md) | 动作张量注入阶段的进展记录。加性 bias 与 FiLM 的失效诊断在这里 |
| [`action_injection_design.md`](action_injection_design.md) | 注入方案的完整技术分析与推导，H3 的三条架构约束 |
| [`minimax_h3_abot_progress.md`](minimax_h3_abot_progress.md) | ABot 数据线的早期进展 |

这两条线的结论已经写进 `pipeline_text_injection.md` 和 `action_text_injection_plan.html`，
读那两篇即可；这里保留原文是因为里面的**实测数字与诊断过程**没有别处记录。

## 背景资料（原始 bundle 随附）

| 文档 | 说明 |
|---|---|
| [`BUNDLE_README.md`](BUNDLE_README.md) | 原 bundle 的设计论证、判据、踩坑清单 |
| [`minimax_h3_architecture_and_data.md`](minimax_h3_architecture_and_data.md) | H3 架构与数据格式 |
| [`minimax_h3_world_model.md`](minimax_h3_world_model.md) | 世界模型方向的调研 |
| [`minimax_h3_abot_data.md`](minimax_h3_abot_data.md) | ABot 数据集说明 |
| [`minimax_h3_vrising_finetune.md`](minimax_h3_vrising_finetune.md) | V-Rising 数据上的微调（前一阶段） |
| [`minimax_h3_vrising_system_overview.md`](minimax_h3_vrising_system_overview.md) | V-Rising 线的系统概览 |
| [`BUNDLE_MANIFEST.txt`](BUNDLE_MANIFEST.txt) | bundle 文件清单（735 KB，校验用） |

---

## 读的顺序

**想知道现在做什么** → `pipeline_text_injection.md`

**想知道这一路是怎么走过来的** → `journey.md`

**想知道为什么这么做** → `action_text_injection_plan.html` → 若想看被否掉的方案，
再翻 `injection_options.html`

**想知道之前为什么失败** → `PROGRESS.md` 的 §4（失效诊断）与 `action_injection_design.md`

**想动手复现** → 仓库根目录 `README.md` 的「复现」一节 + `workspace_layout.md` 的环境部分

---

## 可视化页（在线版）

三个核对台内嵌了视频（每个 3–6 MB），**不入库** —— 它们是 `code/abot/build_*_viz.py`
的生成物，每重建一次就会往 git 历史里塞一个全新的大 blob。在线版：

| 页面 | 比什么 | 链接 |
|---|---|---|
| 推理结果核对台 | 生成 vs 真实片段 | https://claude.ai/code/artifact/e281885a-2132-46ba-8dc0-9617db13b66b |
| 换按键对照台 | 生成 vs 另一个生成（同首帧换按键） | https://claude.ai/code/artifact/e84b3ed3-37aa-42b7-81ab-02a536bc2d8a |
| 长视频实测对照台 | 一次生成 10s vs 分块续写 15.5s + 逐帧变化量曲线 | https://claude.ai/code/artifact/37cfff37-0311-4ea8-a1b6-5e9194d991c3 |
| 架构图 | 注入机制（纯 SVG，已入库） | https://claude.ai/code/artifact/417e81bd-93c8-4a6b-a899-c796d424b737 |

重建方式：

```bash
# 长视频实测对照台
bash code/abot/run_chunked_continue.sh 3
python3 code/abot/build_longvid_viz.py

# 推理结果核对台（跑完 8 卡推理会自动生成）
bash code/abot/run_infer8_text.sh 0 1 2 3 4 5 6 7

# 换按键对照台
bash code/abot/run_action_ab8.sh
python3 code/abot/build_action_ab_viz.py --tag ab8_step9840

# 每次改完页面都要过这一关（假 DOM 下真跑一遍 script）
python3 code/abot/check_page_js.py docs/<页面>.html
```
