#!/usr/bin/env python3
"""动作 -> 逐 latent 文本标注的规则表。

docs/action_text_injection_plan.html「信号设计」一节的实现。手写规则，不过 VLM ——
确定、可复现、零成本，而且推理时能实时生成（流式阶段的硬要求）。

模板是固定句式，只换槽内容：

    the man <怎么动>, camera <相机视角怎么动>

**「相机视角怎么动」用 COLMAP 实测的 d_yaw / d_pitch 生成，不用 IJKL 按键。**
这是整个方案唯一能真正增加信息量的地方：三个从句若全由同样 8 个 bit 推出，
就只是把 one-hot 换成了英文（正是 Incantation 的 Action-Index 基线在做的事）。

阈值来自 7872 训练集前 600 条 clip 的实测分位（池化到 latent 之后）：

    d_yaw   |v| p50=0.145  p70=0.526  p85=0.945  p95=1.076  (ROT_CLIP 截在 3.0)
    d_pitch |v| p50=0.006  p70=0.020  p85=0.229  p95=0.710

符号约定已由 viz_action.py 实证：J = 左转 -> yaw 为正，L = 右转 -> yaw 为负。

自检:
    python3 code/abot/action_script.py --self-test
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import abot_action as A

# 主体锚点：全局一条，让逐帧标注里的 "the man" 有指代物。
# 具体长相由首帧的 <Picture 1> 视觉 token 承载，这里只需要一个稳定的指代。
SUBJECT_ANCHOR = "A third-person view of a man."

# 移动从句。按 W/S/A/D 的固定次序拼接，保证同一组合永远得到同一串
# —— 这是逐条独立编码 + 去重字典能生效的前提。
MOTION = {
    "W": "walks forward",
    "S": "walks backward",
    "A": "strafes left",
    "D": "strafes right",
}
MOTION_ORDER = ("W", "S", "A", "D")
MOTION_IDLE = "stands still"

# 相机从句阈值。三档而不是连续值，是因为语言表达不了连续量，
# 硬要写 "pans left by 0.37" 只会制造词表噪声。
# 阈值定义在**逐帧速率**上，不是 bin 求和值上。
# bin_to_latent 对相机通道是逐帧求和，而 _FRAME_PER_TOKEN=(1,4,4,4,4) 的 bin 宽不等，
# 每第 5 步只覆盖 1 帧、求和值天然只有邻居的 1/4。直接拿求和值判档，会让 8/37 的步
# 被误判成"相机变慢了"——那是 bin 变窄，不是相机变慢。必须先除以 bin 帧数。
# 数值沿用原先在 4 帧 bin 上校准好的那套（0.10 / 0.90 / 0.20 / 0.30）÷ 4。
YAW_MIN, YAW_SHARP = 0.025, 0.225
PITCH_MIN = 0.05
TRANS_MIN = 0.075

# ---- 9 键动作表示 ----------------------------------------------------------
# 原始录像只有 8 个键，第 9 位 F(fast) 是从 COLMAP 实测摇镜速率反推合成的。
# 这么做的理由：方向能从 IJKL 读出（命中率 0.85-0.97），但**速度读不出来**
# —— 按 J 的步里 66% slowly / 34% sharply，接近抛硬币；按住时长也给不出档位
# （第 0 步偏慢只是分箱边界效应，之后就平台了）；ICC≈0.475 说明也不是片段级常量。
# 丢掉速度档，相机从句就退化成 IJKL 的确定性函数，COLMAP 那 6 个通道白测了，
# 方案又回到"把 one-hot 换成英文"。多这一个 bit，众数查表准确率 0.700 -> 0.871。
FAST_COL = "F"
KEYS9 = A.ACTIVE_KEY_COLS + [FAST_COL]
PAN_KEY = {"J": "left", "L": "right"}
TILT_KEY = {"I": "tilts down", "K": "tilts up"}   # 实测 K 的 d_pitch 95% 为正、I 95% 为负
CAMERA_IDLE = "holds steady"
CAMERA_FOLLOW = "follows him"

FAST_COL = "F"
KEYS9 = A.ACTIVE_KEY_COLS + [FAST_COL]
PAN_KEY = {"J": "left", "L": "right"}
TILT_KEY = {"I": "tilts down", "K": "tilts up"}   # 实测 K 的 d_pitch 95% 为正、I 95% 为负
CAMERA_IDLE = "holds steady"
CAMERA_FOLLOW = "follows him"

# 平移方向词。只在"旋转近零且角色没按移动键"时才用 —— 那种情况下相机的平移
# 是独立于按键的新信息；角色在动时相机平移只是跟随，写成 follows him 即可，
# 再详细写就是把 WASD 换个说法重复一遍，白白拉长标注和词表。
TRANS_WORD = {
    "d_x_right": ("drifts right", "drifts left"),
    "d_z_fwd":   ("drifts forward", "drifts back"),
    "d_y_down":  ("drifts down", "drifts up"),
}


def _camera_rate(pooled: np.ndarray, min_frames: int = 4) -> np.ndarray:
    """把逐帧求和的相机通道换算成逐帧速率，并给窄 bin 借邻居凑够窗口。

    两层偏差都来自 _FRAME_PER_TOKEN=(1,4,4,4,4) 的不等宽分箱：
      1. 尺度 —— 1 帧 bin 的求和值天然只有 4 帧 bin 的 1/4，直接判档会误判成"变慢了"
      2. 方差 —— 单帧只覆盖 41 ms，估计更抖，更容易偶然掉到阈值下形成 A-B-A 抖动
    除以 bin 帧数解决第 1 层；窄于 min_frames 的 bin 与后一个 bin 合并估计解决第 2 层。
    合并只影响这一步用于**判档**的速率，不改动 pooled 本身。
    """
    spans = np.array([e - s for s, e in A.frame_spans(len(pooled))], dtype=np.float32)
    raw = pooled[:, A.NUM_KEYS:]
    rate = raw / spans[:, None]
    for k in range(len(pooled)):
        if spans[k] >= min_frames:
            continue
        j = k + 1 if k + 1 < len(pooled) else k - 1
        rate[k] = (raw[k] + raw[j]) / (spans[k] + spans[j])
    return rate


def _motion_clause(keys: np.ndarray) -> str:
    """keys: [8] 的 ACTIVE_KEY_COLS 顺序（W A S D I J K L），只看前四个移动键。

    先做对轴净化：bin_to_latent 对按键取 amax，一个 4 帧窗口里先按 W 后按 S，
    两位就会同时为 1，直接拼串会得到 "walks forward and walks backward" 这种
    自相矛盾的标注。W∧S 与 A∧D 是物理上互斥的对轴，同时置位时相消。
    """
    idx = {name: A.ACTIVE_KEY_COLS.index(name) for name in MOTION_ORDER}
    on = {name: keys[idx[name]] > 0 for name in MOTION_ORDER}
    for a, b in (("W", "S"), ("A", "D")):
        if on[a] and on[b]:
            on[a] = on[b] = False
    words = [MOTION[name] for name in MOTION_ORDER if on[name]]
    if not words:
        return MOTION_IDLE
    return " and ".join(words)


def keys9(pooled: np.ndarray) -> np.ndarray:
    """[latent_t, 17] -> [latent_t, 9] 的 0/1 动作张量。

    前 8 位是原始按键（amax 池化的结果），第 9 位 F 由 COLMAP 实测摇镜速率导出：
    按了摇镜键且逐帧 |d_yaw| >= YAW_SHARP 时置 1。推理时这一位由用户直接按。
    """
    keys = (pooled[:, A.ACTIVE_KEY_INDICES] > 0).astype(np.float32)
    rate = _camera_rate(pooled)
    ji, li = A.ACTIVE_KEY_COLS.index("J"), A.ACTIVE_KEY_COLS.index("L")
    panning = (keys[:, ji] > 0) | (keys[:, li] > 0)
    fast = panning & (np.abs(rate[:, 1]) >= YAW_SHARP)
    return np.concatenate([keys, fast.astype(np.float32)[:, None]], axis=1)


def _purify(on: dict[str, bool], pairs) -> None:
    """对轴互斥的两个键同时置位时相消。

    bin_to_latent 对按键取 amax，一个 4 帧窗口里先按 W 后按 S，两位就会同时为 1，
    直接拼串会得到 "walks forward and walks backward" 这种自相矛盾的标注。
    """
    for a, b in pairs:
        if on[a] and on[b]:
            on[a] = on[b] = False


def _motion_clause(on: dict[str, bool]) -> str:
    _purify(on, (("W", "S"), ("A", "D")))
    words = [MOTION[name] for name in MOTION_ORDER if on[name]]
    return " and ".join(words) if words else MOTION_IDLE


def _camera_clause(on: dict[str, bool], moving: bool) -> str:
    _purify(on, (("J", "L"), ("I", "K")))
    parts = []
    for key, side in PAN_KEY.items():
        if on[key]:
            parts.append(f"pans {side} {'sharply' if on[FAST_COL] else 'slowly'}")
    for key, word in TILT_KEY.items():
        if on[key]:
            parts.append(word)
    if parts:
        return " and ".join(parts)
    return CAMERA_FOLLOW if moving else CAMERA_IDLE


def annotate_from_keys9(k9: np.ndarray) -> list[str]:
    """**标注是这 9 位的纯函数** —— 训练与推理因此走完全同一条路径。

    代价是丢掉了 drifts 那一类（角色没动而相机自己在平移，占 1.7%）：
    它按定义就无法从按键推出，推理时也给不出来，所以并进 holds steady。
    """
    if k9.ndim != 2 or k9.shape[1] != len(KEYS9):
        raise ValueError(f"期望 [latent_t, {len(KEYS9)}]，得到 {list(k9.shape)}")
    out = []
    for row in k9:
        on = {c: bool(row[i] > 0) for i, c in enumerate(KEYS9)}
        motion = _motion_clause(dict(on))
        camera = _camera_clause(dict(on), motion != MOTION_IDLE)
        out.append(f"the man {motion}, camera {camera}")
    return out


def annotate(pooled: np.ndarray, **_ignored) -> list[str]:
    """[latent_t, 17] -> 标注串。先导出 9 位动作，再走纯函数。"""
    return annotate_from_keys9(keys9(pooled))


def script_for_clip(action_path: Path, num_frames: int = 124, latent_t: int = 37,
                    **kw) -> list[str]:
    matrix = np.load(action_path, allow_pickle=False)[:num_frames]
    return annotate(A.bin_to_latent(matrix, latent_t), **kw)


def null_script(latent_t: int = 37) -> list[str]:
    """CFG 的「动作零参考」负 prompt：同样的句式，动作全部置为静止。

    这样 cfg_scale 放大的恰好是动作那部分的差量，而不是笼统的 prompt 服从度。
    """
    return [f"the man {MOTION_IDLE}, camera {CAMERA_IDLE}"] * latent_t


# --------------------------------------------------------------------------- #
def _self_test() -> None:
    import json

    root = Path(__file__).resolve().parents[2]
    rows = [json.loads(l) for l in open(root / "data/abot_meta_train_7872.jsonl")][:400]
    clips = root / "data/clips"

    scripts = [script_for_clip(clips / r["action"]) for r in rows]

    print("== 单条样例 ==")
    print(f"主体锚点: {SUBJECT_ANCHOR}")
    for s in scripts[0][:6]:
        print(f"  {s}")
    print("  …")

    flat = [s for sc in scripts for s in sc]
    vocab = sorted(set(flat))
    print(f"\n== 词表 ==")
    print(f"  {len(rows)} 条 clip × 37 步 = {len(flat)} 条标注，去重后 {len(vocab)} 种")
    print(f"  单条 clip 内平均去重后 {np.mean([len(set(sc)) for sc in scripts]):.1f} 种")

    from collections import Counter
    c = Counter(flat)
    print(f"\n== 出现最多的 8 种 ==")
    for s, n in c.most_common(8):
        print(f"  {n / len(flat):.3f}  {s}")

    idle = sum(1 for s in flat if f"{MOTION_IDLE}, camera {CAMERA_IDLE}" in s)
    print(f"\n== 覆盖度 ==")
    print(f"  完全静止（移动和相机都无）的步: {idle / len(flat):.3f}")
    print(f"  含相机运动的步:               "
          f"{sum(1 for s in flat if CAMERA_IDLE not in s) / len(flat):.3f}")
    print(f"  含移动的步:                   "
          f"{sum(1 for s in flat if MOTION_IDLE not in s) / len(flat):.3f}")

    lens = [len(s) for s in vocab]
    print(f"\n  标注串长度 {min(lens)}–{max(lens)} 字符")
    print(f"\n负 prompt 参考: {null_script(1)[0]}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
