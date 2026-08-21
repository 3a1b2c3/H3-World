#!/usr/bin/env python3
"""V-Rising 动作 sidecar -> MiniMax-H3 latent 时间轴上的按键张量。

这个模块只解决一件事：**把逐帧按键正确地对齐到 H3 非均匀的 latent 时间轴上。**

为什么不能照搬 ReactiveGWM 的 F.adaptive_max_pool1d(keyboard, output_size=f)：
H3 的 video VAE 时间分组是非均匀的，_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)，
每组 5 个 latent token 覆盖 17 个原始帧——第一个只覆盖 1 帧，后 4 个各覆盖 4 帧。
均匀池化会把本该 1:1 对应单帧的那个 token 抹成约 3.34 帧的窗口。按键是稀疏的
二值脉冲，这种错位直接损坏控制信号的时序对齐，而且是系统性的、每组都错。

校验：num_frames=107 时 latent_t = ((107-5)//17)*5+2 = 32，
      而 spans 累加 = 6*17 + (1+4) = 107 ✓（见 self-test）
"""
import itertools
import json
import os

import torch

# H3 video VAE 的时间分组约定，与 MiniMaxH3Unit_PackedSequenceBuilder 里的同名常量一致
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_T_GROUP = 5

# processed_data/dataset_info.json 的 12 键 schema，顺序不能改
BUTTON_COLS = ["W", "A", "S", "D", "Mouse0", "Space", "LeftControl",
               "LeftShift", "Q", "Alpha1", "Alpha2", "Alpha3"]
NUM_BUTTONS = len(BUTTON_COLS)
_COL_INDEX = {k: i for i, k in enumerate(BUTTON_COLS)}

# LeftControl 本身不区分变成哪个形态，按目标形态消歧成 Alpha1/2/3。
# 这是语义必需而非可选装饰：没有它模型只能学到「按 LeftControl 会变成某种东西」，
# 学不到「按哪个键变成哪个形态」的因果。
FORWARD_CTRL_REMAP = {
    "vampire_to_wolf": "Alpha1",
    "vampire_to_bear": "Alpha2",
    "vampire_to_rat": "Alpha3",
}


def latent_t_for(num_frames: int) -> int:
    """H3 的 latent 时间长度。与 train.py 的 17n+5 约束配套。"""
    return ((num_frames - 5) // 17) * _T_GROUP + 2


def frame_spans(latent_t: int):
    """每个 latent token 覆盖的原始帧区间 [start, end)。"""
    spans = [_FRAME_PER_TOKEN[k % _T_GROUP] for k in range(latent_t)]
    ends = list(itertools.accumulate(spans))
    return list(zip([0] + ends[:-1], ends))


def load_sidecar_frames(path: str):
    """读 clip_XXXXXX.json 的逐帧数组。

    sidecar 有两种结构且必须都认：静态形态类直接是 list[81]，技能/变身类是
    {metadata, frames[81]}。早期抽样脚本只认 list、静默跳过 dict，曾据此误判
    「Space / LeftControl 缺失」，实际数据是完整的。
    """
    with open(path) as f:
        d = json.load(f)
    return d if isinstance(d, list) else d["frames"]


def frames_to_matrix(frames, category: str = None) -> torch.Tensor:
    """逐帧记录 -> [T_raw, 12] 的 0/1 矩阵。"""
    remap_to = None
    if category:
        folder = category.split("transform_", 1)[-1]
        remap_to = FORWARD_CTRL_REMAP.get(folder)

    mat = torch.zeros(len(frames), NUM_BUTTONS, dtype=torch.float32)
    for t, fr in enumerate(frames):
        keys = list(fr.get("active_keys") or [])
        # click_events 里的鼠标事件也算按下；schema 里只有 Mouse0
        for ev in (fr.get("click_events") or []):
            btn = ev.get("button") if isinstance(ev, dict) else ev
            if btn is not None:
                keys.append(str(btn))
        for k in keys:
            if k == "LeftControl" and remap_to is not None:
                # 消歧后仍保留 LeftControl 本身：它确实被按下了
                mat[t, _COL_INDEX[remap_to]] = 1.0
            idx = _COL_INDEX.get(k)
            if idx is not None:          # F/E/R/Mouse1/Tab 不在 schema，丢弃
                mat[t, idx] = 1.0
    return mat


def resample_to_video_timeline(mat: torch.Tensor, num_frames: int,
                               raw_fps: float, target_fps: float = 24.0) -> torch.Tensor:
    """[T_raw, 12] @raw_fps -> [num_frames, 12] @target_fps。

    **这一步不能省。** sidecar 是 81 帧 @16fps，而视频被 fix_frame_rate=True 重采样到
    24fps 后取 107 帧。直接拿 81 帧当 107 帧的时间轴分箱，会把动作时间压缩 1.32 倍，
    是整段系统性错位。

    映射精确复刻 core/data/operators.py 的 FrameSamplerByRateMixin.map_single_frame_id，
    与视频帧的取法逐帧一致，不是近似。
    """
    j = torch.arange(num_frames, dtype=torch.float64)
    raw_idx = torch.round(j / target_fps * raw_fps).long().clamp(max=mat.shape[0] - 1)
    return mat[raw_idx]


def infer_raw_fps(frames, default: float = 16.0) -> float:
    """从 sidecar 的 timestamp 反推原始帧率。"""
    if len(frames) < 2:
        return default
    t0, t1 = frames[0].get("timestamp"), frames[-1].get("timestamp")
    if t0 is None or t1 is None or t1 <= t0:
        return default
    return (len(frames) - 1) / (t1 - t0)


def bin_to_latent(mat: torch.Tensor, latent_t: int) -> torch.Tensor:
    """[num_frames, 12] -> [latent_t, 12]，按 H3 的非均匀分组做窗口 amax。

    用 amax 而不是 mean：按键是二值脉冲，窗口内按下过就算按下。取均值会把
    单帧脉冲稀释成 0.25 这种没有物理含义的强度。
    """
    spans = frame_spans(latent_t)
    need = spans[-1][1]
    if mat.shape[0] < need:              # 末尾不足则重复最后一帧补齐
        pad = mat[-1:].expand(need - mat.shape[0], -1)
        mat = torch.cat([mat, pad], dim=0)
    return torch.stack([mat[s:e].amax(0) for s, e in spans])


def action_from_sidecar(path: str, num_frames: int = 107, category: str = None) -> torch.Tensor:
    """sidecar 路径 -> [latent_t, 12]，一步到位（含重采样）。"""
    frames = load_sidecar_frames(path)
    mat = frames_to_matrix(frames, category)
    mat = resample_to_video_timeline(mat, num_frames, infer_raw_fps(frames))
    return bin_to_latent(mat, latent_t_for(num_frames))


def sidecar_path_for(video_rel: str, data_base: str) -> str:
    """metadata 的 video 字段（如 wolf/clip_000001.mp4）-> 同名 .json。"""
    return os.path.join(data_base, os.path.splitext(video_rel)[0] + ".json")


if __name__ == "__main__":
    # 自检：帧映射必须精确复现 num_frames，否则整条动作时间轴都是错的
    for nf in (107, 90, 73, 56, 39, 22):
        lt = latent_t_for(nf)
        spans = frame_spans(lt)
        assert spans[-1][1] == nf, f"num_frames={nf}: spans 累加 {spans[-1][1]} != {nf}"
        print(f"num_frames={nf:4d}  latent_t={lt:3d}  spans 累加={spans[-1][1]:4d} ✓")

    print("\n107 帧时前 10 个 token 的帧区间:", frame_spans(32)[:10])
    print("最后 5 个:", frame_spans(32)[-5:])
