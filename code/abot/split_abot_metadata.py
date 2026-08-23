#!/usr/bin/env python3
"""Create a deterministic ABot train/test split while preserving row order."""

import argparse
import hashlib
import json
import os

import numpy as np

import abot_action as A


def stable_score(sample_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()


def write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def action_coverage(rows: list[dict], clips_dir: str) -> dict[str, int]:
    counts = {name: 0 for name in A.ACTIVE_KEY_COLS}
    for row in rows:
        mat = np.load(os.path.join(clips_dir, row["action"]), mmap_mode="r")
        for name, index in zip(A.ACTIVE_KEY_COLS, A.ACTIVE_KEY_INDICES):
            counts[name] += int(np.any(mat[:124, index] > 0))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument(
        "--fixed-test",
        default=None,
        help="沿用已有测试集：从这个 jsonl 读 sample_id，把它们全部划为测试、其余为训练。"
             "扩数据时必须用它 —— hash 选法只对固定的输入集合稳定，输入从 16000 变成 "
             "20128 之后同一个 seed 会选出**不同的** 128 条，导致新训练集里混进旧测试样本。",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not 0 < args.test_size < len(rows):
        parser.error("--test-size must be between 1 and len(input)-1")

    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("sample_id is not unique; refusing an episode-leaking split")

    if args.fixed_test:
        with open(args.fixed_test, encoding="utf-8") as handle:
            keep = {json.loads(line)["sample_id"] for line in handle if line.strip()}
        missing = keep - set(ids)
        if missing:
            raise SystemExit(
                f"--fixed-test 里有 {len(missing)} 条不在 input 中，例: {sorted(missing)[:3]}")
        test_indices = {i for i, sid in enumerate(ids) if sid in keep}
        print(f"沿用固定测试集 {args.fixed_test}：{len(test_indices)} 条")
    else:
        test_indices = set(sorted(range(len(rows)),
                                  key=lambda i: stable_score(ids[i], args.seed))[:args.test_size])
    train_rows = [row for i, row in enumerate(rows) if i not in test_indices]
    test_rows = [row for i, row in enumerate(rows) if i in test_indices]

    for row in rows:
        for key in ("video", "action"):
            path = os.path.join(args.clips_dir, row[key])
            if not os.path.isfile(path):
                raise SystemExit(f"missing {key}: {path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.train_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.test_output)), exist_ok=True)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.test_output, test_rows)

    print(f"input={len(rows)} train={len(train_rows)} test={len(test_rows)} seed={args.seed}")
    print("test action coverage:")
    for name, count in action_coverage(test_rows, args.clips_dir).items():
        print(f"  {name}: {count}/{len(test_rows)} clips")
        if count == 0:
            raise SystemExit(f"test split does not cover action {name}")


if __name__ == "__main__":
    main()
