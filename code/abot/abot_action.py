#!/usr/bin/env python3
"""ABot-World-Explorer 动作标注 -> MiniMax-H3 latent 时间轴上的条件张量。

与 V-Rising 的 `/nfs/danze/h3_action.py` 是同一件事的另一个数据源版本，
但有三处本质区别，都是实测出来的（见 minimax_h3_abot_data.md §3）：

1. **`action.json` 里的 delta_* 四个向量全是 0**（296 条抽样 100% 为零），
   连续运动信号必须自己从 COLMAP 的 `sparse/0/images.txt` 反推。
   好消息是 pose 覆盖率实测 1800/1800 = 100%，逐帧都有。

2. **COLMAP 的尺度是逐 episode 任意的**：同样只按 W 前进，
   每帧位移中位数在 40 条抽样里横跨 0.0045 ~ 0.1488（33 倍）。
   所以平移通道必须**按 episode 归一化**；旋转是角度、天然无尺度问题。

3. **不需要 resample_to_video_timeline。** V-Rising 那边 sidecar 是 81 帧 @16fps
   而视频被重采样到 107 帧 @24fps，必须补一步重采样。这里我们在切片时就
   用显式帧号列表同时裁视频和取动作，两者**按构造逐帧对齐**，
   `LoadVideo` 读回来是恒等映射（已实测 0 重复帧）。

通道布局（`ACTION_DIM = 17`）：

    0..10   11 个按键，二值，窗口内 amax
    11..13  旋转增量 pitch/yaw/roll，单位度，窗口内 sum，再 / ROT_SCALE
    14..16  平移增量 (x右, y下, z前)，前一帧相机系，已按 episode 归一化，
            窗口内 sum，再 / TRA_SCALE

为什么二值用 amax、连续用 sum：按键是脉冲，窗口内按下过就算按下，取均值会
稀释成 0.25 这种没有物理含义的强度；而增量是可加的物理量，一个 latent token
覆盖 4 帧就该拿到这 4 帧的**总位移**。注意 H3 的分组是非均匀的 (1,4,4,4,4)，
每组第一个 token 只覆盖 1 帧，所以它的 sum 天然只有后面的 1/4 —— 这是真实信息
（那个 token 确实只代表 1 帧），不是需要抹平的假象。
"""
import itertools
import json
import tarfile

import numpy as np

# ---- H3 video VAE 的时间分组约定，与 h3_action.py / PackedSequenceBuilder 一致 ----
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_T_GROUP = 5

# ---- 数据集原生 schema。control_scheme = WASD_QE_locomotion_IJKL_rotation ----
# 40 条抽样里 Q/E/Space 从未按下，但只抽了 296/30969，不能断言全集为零，
# 而且保留 3 个恒零通道的代价只是 3*5376*50 个拿不到梯度的参数，
# 换来的是 schema 与数据集原生定义逐位一致。宁可留着。
KEY_COLS = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"]
ACTIVE_KEY_COLS = ["W", "A", "S", "D", "I", "J", "K", "L"]
ACTIVE_KEY_INDICES = [KEY_COLS.index(name) for name in ACTIVE_KEY_COLS]
ROT_COLS = ["d_pitch", "d_yaw", "d_roll"]
TRA_COLS = ["d_x_right", "d_y_down", "d_z_fwd"]
ACTION_COLS = KEY_COLS + ROT_COLS + TRA_COLS
NUM_KEYS = len(KEY_COLS)
ACTION_DIM = len(ACTION_COLS)

# 把 sum 后的量级压到 O(1)。实测（40 条 episode，每 4 帧一箱）：
# 旋转 p95 中位 2.24 度、max 中位 3.73；归一化平移 p95 约 4（每帧 ~1）。
ROT_SCALE = 4.0
TRA_SCALE = 4.0

# 30fps -> 24fps：每 5 个源帧丢第 5 个。选这条规则而不是复刻 H3 的 round()
# 映射，是因为它能被 ffmpeg 的 select 滤镜精确表达、也能精确求逆，
# 于是「视频帧 j 对应哪个源帧」由我们说了算，不依赖对 H3 内部实现的推断。
_KEEP_OF, _DROP_EVERY = 4, 5


def window_offsets(n_out: int):
    """输出帧 j -> 相对窗口起点的源帧偏移。"""
    return [(j // _KEEP_OF) * _DROP_EVERY + (j % _KEEP_OF) for j in range(n_out)]


def window_span(n_out: int) -> int:
    """一个 n_out 帧的输出窗口要吃掉多少源帧。"""
    return window_offsets(n_out)[-1] + 1


def latent_t_for(num_frames: int) -> int:
    return ((num_frames - 5) // 17) * _T_GROUP + 2


def frame_spans(latent_t: int):
    spans = [_FRAME_PER_TOKEN[k % _T_GROUP] for k in range(latent_t)]
    ends = list(itertools.accumulate(spans))
    return list(zip([0] + ends[:-1], ends))


# --------------------------------------------------------------------------- #
# COLMAP
# --------------------------------------------------------------------------- #
def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Hamilton (w,x,y,z) -> 3x3。COLMAP 存的是 world->camera 的 R_cw。"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _parse_images_txt(blob: bytes):
    """images.txt -> {NAME(无扩展名): (quat[4], t_cw[3])}。

    只取每条记录的第一行；第二行是 2D 观测，这个数据集里 points3D.txt 是空的，
    观测行也没有用处。IMAGE_ID 不保证连续，所以一律以 NAME 为键。
    """
    out = {}
    for line in blob.decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 10:
            continue                      # 观测行
        try:
            quat = np.array([float(v) for v in p[1:5]])
            tvec = np.array([float(v) for v in p[5:8]])
        except ValueError:
            continue                      # 观测行恰好够长时的兜底
        out[p[-1].rsplit(".", 1)[0]] = (quat, tvec)
    return out


def read_episode(tar_path: str):
    """读一个 annotations.tar，返回逐源帧的按键与位姿。

    返回 dict:
        keys     [T, 11] float32   0/1
        R_cw     [T, 3, 3]
        C_world  [T, 3]            相机中心 = -R_cw^T t_cw
        fps, total_frames, control_scheme
    只有 action.json 与 images.txt 都覆盖到的帧才会被保留，且顺序按
    action.json 的 frame_id 排列（实测 NAME 与 frame_id 逐条对得上）。
    """
    with tarfile.open(tar_path) as t:
        members = {m.name: m for m in t.getmembers()}
        allow = {"action.json", "caption.json",
                 "sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt"}
        bad = [n for n, m in members.items() if n not in allow or not m.isfile()]
        if bad:
            raise ValueError(f"{tar_path}: 非预期的 tar 成员 {bad[:3]}")
        act = json.load(t.extractfile("action.json"))
        poses = _parse_images_txt(t.extractfile("sparse/0/images.txt").read())
        cap = json.load(t.extractfile("caption.json"))

    frames = act["frames"]
    order = [f["frame_id"] for f in frames]
    keep = [i for i, n in enumerate(order) if n in poses]
    if len(keep) != len(order):
        # 文档明确说「不要假设每帧都有 pose」。实测 40/40 是满覆盖，
        # 但真遇到缺帧时宁可整条丢弃，也不要静默补零造出假的静止动作。
        raise ValueError(f"{tar_path}: pose 覆盖 {len(keep)}/{len(order)}，不是满覆盖")

    keys = np.zeros((len(frames), NUM_KEYS), dtype=np.float32)
    for i, fr in enumerate(frames):
        k = fr["keys"]
        for c, name in enumerate(KEY_COLS):
            if k.get(name):
                keys[i, c] = 1.0

    R = np.stack([quat_to_R(poses[n][0]) for n in order])
    T = np.stack([poses[n][1] for n in order])
    C = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -T)
    return dict(keys=keys, R_cw=R, C_world=C,
                fps=float(act.get("fps", 30.0)),
                total_frames=len(frames),
                control_scheme=act.get("control_scheme"),
                caption=cap)


def pose_deltas(R: np.ndarray, C: np.ndarray):
    """位姿序列 -> 逐步增量。第 0 行恒为 0（没有「上一帧」）。

    平移表达在**上一帧的相机系**里（+x 右 / +y 下 / +z 前，COLMAP 约定），
    这样 W 前进就落在 +z 上，与按键语义直接对应，且与世界系朝向无关。
    旋转取相对旋转 R_rel = R[i] @ R[i-1]^T 再拆成 pitch/yaw/roll（度）。

    注意增量是在**已经抽帧之后**的序列上算的，不是先算逐源帧增量再求和 ——
    这样跨过被丢弃的源帧时结果依然精确，不依赖小角度可加的近似。
    """
    n = len(C)
    dC = np.zeros((n, 3), dtype=np.float64)
    dE = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return dC, dE
    dC[1:] = np.einsum("nij,nj->ni", R[:-1], np.diff(C, axis=0))
    Rrel = np.einsum("nij,njk->nik", R[1:], R[:-1].transpose(0, 2, 1))
    dE[1:, 0] = np.degrees(np.arcsin(np.clip(-Rrel[:, 1, 2], -1.0, 1.0)))   # pitch
    dE[1:, 1] = np.degrees(np.arctan2(Rrel[:, 0, 2], Rrel[:, 2, 2]))        # yaw
    dE[1:, 2] = np.degrees(np.arctan2(Rrel[:, 1, 0], Rrel[:, 1, 1]))        # roll
    return dC, dE


def episode_translation_scale(ep: dict, n_probe_out: int = 1400) -> float:
    """这条 episode 的「典型每输出帧位移」，用来消掉 COLMAP 的任意尺度。

    在**抽帧之后**的时间轴上算，这样单位就是「每个 24fps 帧」，与训练时一致。
    取中位数而不是均值：中位数不被少数重建跳变主导。
    """
    idx = np.array(window_offsets(n_probe_out))
    idx = idx[idx < ep["total_frames"]]
    dC, _ = pose_deltas(ep["R_cw"][idx], ep["C_world"][idx])
    spd = np.linalg.norm(dC[1:], axis=1)
    med = float(np.median(spd)) if len(spd) else 0.0
    return med


# --------------------------------------------------------------------------- #
# 切窗口 -> 逐帧动作矩阵
# --------------------------------------------------------------------------- #
def window_action_matrix(ep: dict, start: int, n_out: int, scale: float) -> np.ndarray:
    """[n_out, 17]，与切出来的 mp4 逐帧对齐（输出帧 j <-> 源帧 start+offsets[j]）。

    平移已除以 `scale`（episode 级归一化），旋转保持原始度数，
    两者都**还没有**除以 ROT_SCALE / TRA_SCALE —— 那一步留到分箱时做，
    这样想调量纲不必重跑预处理。
    """
    idx = np.array([start + o for o in window_offsets(n_out)])
    if idx[-1] >= ep["total_frames"]:
        raise ValueError(f"窗口越界: 需要源帧 {idx[-1]}，只有 {ep['total_frames']}")
    dC, dE = pose_deltas(ep["R_cw"][idx], ep["C_world"][idx])
    if scale > 1e-9:
        dC = dC / scale
    else:
        dC = np.zeros_like(dC)            # 整条几乎不动，归一化没有意义
    return np.concatenate([ep["keys"][idx], dE, dC], axis=1).astype(np.float32)


# 连续通道的对称截断阈值。**两个都必需** —— 位姿序列里混着重建跳变，
# 不截断的话极少数样本会给零初始化的 embedder 灌进量级失衡的梯度。
#
# 实测 2225 条切片（分箱缩放后）：
#   平移 p99 ≈ 2.4，不截断时 max = 47.5  -> TRA_CLIP=4.0 只切 0.174% 的元素
#   旋转 p99 ≈ 1.2，不截断时 max = 21.2  -> ROT_CLIP=3.0 只切 0.020% / 10 条 clip
#
# 阈值不是拍的：旋转分位数在 p99.9=1.70 与 p99.99=9.59 之间有明显拐点，
# 说明极值是离散离群点而非连续长尾。更强的判据是**这些极值和按键对不上** ——
# yaw>=3.0 的 48 个 token 里只有 15% 同时按着 J/L，真的快速转身不可能不按转向键。
# 所以它们是重建跳变，该切。3.0 = 12 度/箱 = 72 度/秒，真实快速转身仍然完整保留。
TRA_CLIP = 4.0
ROT_CLIP = 3.0


def bin_to_latent(mat: np.ndarray, latent_t: int,
                  rot_scale: float = ROT_SCALE,
                  tra_scale: float = TRA_SCALE,
                  tra_clip: float = TRA_CLIP,
                  rot_clip: float = ROT_CLIP) -> np.ndarray:
    """[num_frames, 17] -> [latent_t, 17]，按 H3 的非均匀分组分箱。

    二值段取 amax，连续段取 sum 再缩放，两个连续段都做对称截断。
    """
    spans = frame_spans(latent_t)
    if spans[-1][1] != len(mat):
        raise ValueError(f"分箱区间累加 {spans[-1][1]} != 动作帧数 {len(mat)}")
    out = np.zeros((latent_t, mat.shape[1]), dtype=np.float32)
    for k, (s, e) in enumerate(spans):
        out[k, :NUM_KEYS] = mat[s:e, :NUM_KEYS].max(axis=0)
        out[k, NUM_KEYS:] = mat[s:e, NUM_KEYS:].sum(axis=0)
    out[:, NUM_KEYS:NUM_KEYS + 3] /= rot_scale
    out[:, NUM_KEYS + 3:] /= tra_scale
    if rot_clip:
        np.clip(out[:, NUM_KEYS:NUM_KEYS + 3], -rot_clip, rot_clip,
                out=out[:, NUM_KEYS:NUM_KEYS + 3])
    if tra_clip:
        np.clip(out[:, NUM_KEYS + 3:], -tra_clip, tra_clip, out=out[:, NUM_KEYS + 3:])
    return out


# --------------------------------------------------------------------------- #
def _self_test():
    print("== 分组自检 ==")
    for nf in (107, 124, 141, 175, 192):
        lt = latent_t_for(nf)
        sp = frame_spans(lt)
        ok = sp[-1][1] == nf
        print(f"  num_frames={nf:4d}  latent_t={lt:3d}  spans累加={sp[-1][1]:4d}  {'OK' if ok else 'FAIL'}")
        assert ok
    print("  124 帧前 8 个 token 区间:", frame_spans(latent_t_for(124))[:8])

    print("\n== 抽帧映射自检 ==")
    for n_out in (124, 130):
        off = window_offsets(n_out)
        assert len(set(off)) == n_out and off == sorted(off)
        print(f"  n_out={n_out}  span={window_span(n_out)} 源帧  "
              f"有效帧率={n_out / (window_span(n_out) / 30.0):.3f} fps  前8={off[:8]}")
    assert abs(124 / (window_span(124) / 30.0) - 24.0) < 0.3

    print("\n== 分箱语义自检 ==")
    m = np.zeros((124, ACTION_DIM), dtype=np.float32)
    m[3, 0] = 1.0                                  # 第 3 帧按下 W(单帧脉冲)
    m[:, NUM_KEYS + 1] = 1.0                       # 每帧 yaw 1 度
    b = bin_to_latent(m, latent_t_for(124))
    # 帧 3 落在 token 1 (区间 [1,5))
    assert b[0, 0] == 0.0 and b[1, 0] == 1.0, b[:3, 0]
    # token0 覆盖 1 帧 -> 1/4 度; token1 覆盖 4 帧 -> 4/4 度
    assert abs(b[0, NUM_KEYS + 1] - 1 / ROT_SCALE) < 1e-6
    assert abs(b[1, NUM_KEYS + 1] - 4 / ROT_SCALE) < 1e-6
    print(f"  单帧脉冲落在 token1 而非被均值稀释: OK")
    print(f"  yaw 每帧1度 -> token0={b[0, NUM_KEYS+1]:.3f} token1={b[1, NUM_KEYS+1]:.3f} "
          f"(1帧 vs 4帧, 比值 4x): OK")

    print("\n== 四元数/位姿自检 ==")
    R = quat_to_R(np.array([1.0, 0, 0, 0]))
    assert np.allclose(R, np.eye(3))
    rng = np.random.default_rng(0)
    q = rng.normal(size=4); R = quat_to_R(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    print("  随机四元数 -> 正交且 det=+1: OK")

    print("\n全部通过。")


if __name__ == "__main__":
    _self_test()
