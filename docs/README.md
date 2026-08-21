# 文档索引

按「当前有效 / 阶段性记录 / 背景资料」分三层。**只想知道现在怎么回事，看前两篇就够。**

## 当前有效

| 文档 | 内容 |
|---|---|
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

**想知道为什么这么做** → `action_text_injection_plan.html` → 若想看被否掉的方案，
再翻 `injection_options.html`

**想知道之前为什么失败** → `PROGRESS.md` 的 §4（失效诊断）与 `action_injection_design.md`

**想动手复现** → 仓库根目录 `README.md` 的「复现」一节 + `workspace_layout.md` 的环境部分
