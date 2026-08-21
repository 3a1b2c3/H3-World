#!/usr/bin/env python3
"""把 v_rising 的 train.jsonl 转成 MiniMax-H3 训练用的 metadata。

要做的事只有两件：
  1. 加一个 input_audio 字段，指向 mp4 本身。H3 是音视频联合模型，训练必须有这个 key，
     否则 train.py 的 parse_extra_inputs 会 KeyError。v_rising 的 clip 要么没音轨、
     要么是静音轨，LoadAudioWithTorchaudio 加载失败会返回 None，再由
     --silent_on_missing_audio 换成静音张量。
  2. 按 kind 拆出子集，方便先小规模验证再全量跑。

输出（都在 transform_classified/ 下）：
  h3_meta_all.jsonl        20699 条，全部
  h3_meta_transform.jsonl   9314 条，形态转换（FL2VA 首尾帧最契合）
  h3_meta_smoke.jsonl          64 条，冒烟测试用
"""
import json
import os
import random

BASE = "/nfs/danze/data/v_rising/vrising_train_bundle/data/transform_classified"
SRC = os.path.join(BASE, "train.jsonl")


def main():
    rows = []
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "video": d["video"],
                "input_audio": d["video"],   # 指向 mp4 自身，无音轨时走静音兜底
                "prompt": d["prompt"],
                "kind": d.get("kind"),
                "category": d.get("category"),
            })

    def dump(name, subset):
        path = os.path.join(BASE, name)
        with open(path, "w") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name:26s} {len(subset):6d} 条")
        return path

    print(f"读入 {SRC}: {len(rows)} 条\n生成:")
    dump("h3_meta_all.jsonl", rows)
    dump("h3_meta_transform.jsonl", [r for r in rows if r["kind"] == "transform"])

    # 按 category 拆窄切片。63M 参数的 rank-32 LoRA 在 1000-10000 步收敛，
    # all(20699) x 5 epoch = 103495 步严重过量，窄切片才落在合理区间。
    # 这里一次导出狼相关的四类：form_wolf 单独用作 Stage B 的通路验证床
    # （动作空间只有 WASD，无形态突变），另外三类留给后续升级到「狼的完整生命周期」。
    for cat in ("form_wolf", "ability_wolf_space",
                "transform_vampire_to_wolf", "transform_wolf_to_vampire"):
        dump(f"h3_meta_{cat}.jsonl", [r for r in rows if r["category"] == cat])

    # 狼的完整生命周期：漫游 + 变身双向。form_wolf 撑不住变身能力时升级到这一档。
    wolf_life = {"form_wolf", "transform_vampire_to_wolf", "transform_wolf_to_vampire"}
    dump("h3_meta_wolf_life.jsonl", [r for r in rows if r["category"] in wolf_life])

    rng = random.Random(0)
    smoke = rng.sample(rows, 64)
    dump("h3_meta_smoke.jsonl", smoke)


if __name__ == "__main__":
    main()
