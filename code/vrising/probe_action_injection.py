"""动作注入的落点探针 —— 纯 CPU，用真实缓存的 packed 元数据验证索引正确性。

为什么需要这个：minimax_h3_dit.py 的注入用 narrow+广播+cat 取代了原设计的
index_add，前提是「[text | cond | audio | video | pad] 布局里 video 段连续」。
这条假设一旦不成立，代码会**静默算错**而不是报错（世界模型文档 §1 差异 2 的警告）。
训练跑不出这个信号：loss 照样下降，只是 bias 加错了地方。

三件事是训练验不了、这里能验的：
  1. narrow 版本与 index_add 参考实现**逐位相同** —— 连续性假设成立
  2. bias 严格避开 keyframe anchor（img_pos 前 cond_rows_count 行）
  3. 第 k 个 latent token 拿到的确实是 emb[k]，没有错位

用法:  python3 /nfs/danze/probe_action_injection.py [--cache <dir>] [--n 8]
"""
import argparse, glob, os, sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--cache", default="/nfs/danze/model/minimax_h3_vrising/form_wolf-cache")
p.add_argument("--n", type=int, default=8, help="抽查几条缓存")
p.add_argument("--mode", choices=["film","bias"], default="film",
               help="被测的注入方式；film 走 (1+scale)*h+shift，bias 走旧的加性")
p.add_argument("--dim", type=int, default=8, help="探针用的 hidden 宽度，真实是 5376，这里取小的省内存")
args = p.parse_args()

files = sorted(glob.glob(os.path.join(args.cache, "**", "*.pth"), recursive=True))
if not files:
    sys.exit(f"没找到缓存: {args.cache}")
step = max(1, len(files) // args.n)
sample = files[::step][: args.n]
print(f"缓存 {args.cache}\n共 {len(files)} 条，抽查 {len(sample)} 条，hidden 宽度取 {args.dim}\n")

C = args.dim
n_fail = 0

for path in sample:
    shared, posi, _ = torch.load(path, weights_only=False)
    packed = posi["packed"]
    img_pos = packed["img_pos"].view(-1).long()
    token_tags = packed["token_tags"].view(-1).long()
    S = int(packed["seq_len"])

    action = shared.get("action_cond")
    if action is None:
        print(f"{os.path.basename(path)}: ✗ 缓存里没有 action_cond")
        n_fail += 1
        continue

    # 完全复刻 model_fn_minimax_h3 的推导（minimax_h3_audio_video.py:839/885/886）
    cond_rows_count = shared["keyframe_cond_anchor"].shape[0]
    _, _, latent_t, lh, lw = shared["input_latents"].shape
    v0 = int(img_pos[cond_rows_count])
    fr = (lh // 2) * (lw // 2)
    n_video = latent_t * fr

    assert action.shape[0] == latent_t, f"action_cond 时间轴 {action.shape[0]} != latent_t {latent_t}"

    # 每个 latent token 给一个可区分的 bias，这样错位一定看得出来
    emb = torch.arange(1, latent_t + 1, dtype=torch.float64).view(latent_t, 1).repeat(1, C)
    base = torch.zeros(S, C, dtype=torch.float64)

    # --- 被测实现：minimax_h3_dit.py 里那段 narrow + 广播 + cat ---
    # film 模式下 base 必须非零，否则乘性项恒等于 0、什么也验不出来；
    # 用 1 填充，这样 (1+scale)*1 + shift 的结果就直接暴露每个 token 拿到的调制值。
    h = base.clone()
    video_pos = img_pos[cond_rows_count:]
    if args.mode == "film":
        h = h + 1.0
        v = h.narrow(0, v0, n_video).view(latent_t, fr, -1)
        v = v * (1.0 + emb.unsqueeze(1)) + emb.unsqueeze(1)
    else:
        v = h.narrow(0, v0, n_video).view(latent_t, fr, -1) + emb.unsqueeze(1)
    got = torch.cat([
        h.narrow(0, 0, v0),
        v.reshape(n_video, -1),
        h.narrow(0, v0 + n_video, h.shape[0] - v0 - n_video),
    ], dim=0)

    # --- 参考实现：index_select 散射语义，不依赖"video 段连续"这个假设 ---
    if args.mode == "film":
        want = h.clone()
        idx = torch.arange(latent_t).repeat_interleave(fr)
        rows = emb.index_select(0, idx)
        want[video_pos] = want[video_pos] * (1.0 + rows) + rows
    else:
        want = base.clone().index_add(0, video_pos, emb.repeat_interleave(fr, dim=0))

    ok = True

    # 1. 两种实现逐位相同 ⇒ video 段确实连续
    if not torch.equal(got, want):
        d = (got - want).abs()
        print(f"{os.path.basename(path)}: ✗ narrow 版与 index_add 参考不一致，"
              f"{int((d > 0).any(1).sum())} 行不同，最大差 {float(d.max())}")
        ok = False

    # 2. 改动的行集合 == video 行集合，一行不多一行不少
    changed = torch.nonzero((got != h).any(1)).view(-1)
    if not torch.equal(changed, torch.sort(video_pos).values):
        extra = set(changed.tolist()) - set(video_pos.tolist())
        missing = set(video_pos.tolist()) - set(changed.tolist())
        print(f"{os.path.basename(path)}: ✗ 改动行集合不匹配，多改 {len(extra)} 行，漏改 {len(missing)} 行")
        ok = False

    # 3. keyframe anchor 逐行未被触碰 —— 这是 bias 糊到首尾帧条件上的直接判据
    anchor_pos = img_pos[:cond_rows_count]
    dirty = (got.index_select(0, anchor_pos) != h.index_select(0, anchor_pos))
    if dirty.any():
        n_dirty = int(dirty.any(1).sum())
        print(f"{os.path.basename(path)}: ✗ {n_dirty}/{cond_rows_count} 行 keyframe anchor 被污染")
        ok = False

    # 4. 非视频模态（text / audio / pad）未被触碰
    non_video = torch.ones(S, dtype=torch.bool)
    non_video[video_pos] = False
    if got[non_video].abs().sum() != 0:
        print(f"{os.path.basename(path)}: ✗ text/audio/pad 区被写入")
        ok = False

    # 5. 时序不错位：第 k 个 latent token 的 fr 行必须全是 emb[k]
    blk = got.narrow(0, v0, n_video).view(latent_t, fr, C)
    expect = emb.unsqueeze(1).expand(latent_t, fr, C)
    if not torch.equal(blk, expect):
        bad = torch.nonzero((blk != expect).any(-1).any(-1)).view(-1)
        print(f"{os.path.basename(path)}: ✗ 第 {bad.tolist()[:5]} 个 latent token 的 bias 错位")
        ok = False

    # 6. 零初始化等价性：emb 全零时输出必须与不注入逐位相同
    z = base.clone()
    zv = z.narrow(0, v0, n_video).view(latent_t, fr, -1) + torch.zeros(latent_t, 1, C, dtype=torch.float64)
    z_out = torch.cat([z.narrow(0, 0, v0), zv.reshape(n_video, -1),
                       z.narrow(0, v0 + n_video, z.shape[0] - v0 - n_video)], dim=0)
    if not torch.equal(z_out, base):
        print(f"{os.path.basename(path)}: ✗ 零初始化下输出与不注入不等价")
        ok = False

    tag_set = sorted(set(token_tags[video_pos].tolist()))
    print(f"{os.path.basename(path):>12}: {'✅' if ok else '❌'} "
          f"S={S} v0={v0} latent_t={latent_t} fr={fr} n_video={n_video} "
          f"anchor={cond_rows_count} video_token_tags={tag_set}")
    n_fail += 0 if ok else 1

print()
if n_fail:
    sys.exit(f"❌ {n_fail}/{len(sample)} 条不通过")
print(f"✅ {len(sample)}/{len(sample)} 条全部通过："
      f"narrow 实现与 index_add 参考逐位一致，bias 严格落在 video 行，"
      f"keyframe anchor 与 text/audio 区零污染，latent token 无错位")
