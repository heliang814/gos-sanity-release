"""从 Phase A baseline (gos_original) 结果里挑 5 个目标 query 给 Phase C 用。

输入  ：<results_dir>/runs.jsonl，仅看 bundle_type == "gos_original" 的行。
输出  ：data/queries_selected.json，与 select_queries.py 输出 schema 兼容
        ([{"id", "query", "bucket"?}, ...])。
规则  ：
  - 只考虑 error_type=None 且 skip_reason=None（"干净跑完"）。
  - reward=1.0 选 3 个、reward=0.0 选 2 个，共 5 个。
  - 在每组内排序：按 ``|runtime - median(runtime)|`` 升序优先（最像中位数的优先），
    token 缺失（None）时降级使用 runtime（不再按 token 排）。
  - 同组内若候选不足，按现有数量返回（脚本仍写 json，留给上层判断要不要继续）。

注意：source query json 既可能来自 select_queries.py 的输出（含 bucket），
也可能是 phase A 改造后存的；都直接用 jsonl 里的 query 文本即可，不依赖外部
queries.json 还原 prompt。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml


def _is_clean(rec: dict) -> bool:
    return rec.get("error_type") is None and rec.get("skip_reason") is None


def _pick(records: list[dict], target_reward: float, k: int) -> list[dict]:
    """从 baseline 行里挑 reward==target_reward 的前 k 个。

    排序键：|runtime - median(runtime)| 升序。
    token_count 缺失时不影响 runtime 排序，仅在 tie-break 时降级（不引入新维度）。
    """
    pool = [
        r for r in records
        if r.get("reward") is not None and float(r["reward"]) == target_reward
    ]
    if not pool:
        return []
    runtimes = [
        float(r.get("execution_time", 0.0)) for r in pool
        if r.get("execution_time") is not None
    ]
    median_rt = statistics.median(runtimes) if runtimes else 0.0

    def _key(r: dict) -> tuple[float, float, str]:
        rt = float(r.get("execution_time") or 0.0)
        # 主键：到中位数的距离
        primary = abs(rt - median_rt)
        # 次键：runtime 本身（在主键并列时偏好更接近中位数偏小的；token 缺失时也 deterministic）
        secondary = rt
        # 末键：query_id 字典序，保证 stable / 可复现
        return (primary, secondary, str(r.get("query_id", "")))

    pool.sort(key=_key)
    return pool[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="data/queries_selected.json")
    ap.add_argument("--reward1-count", type=int, default=3)
    ap.add_argument("--reward0-count", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results_dir = Path(cfg["paths"]["results_dir"]).expanduser().resolve()
    jsonl_path = results_dir / "runs.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"baseline jsonl not found: {jsonl_path}")

    rows: list[dict] = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("bundle_type") != "gos_original":
            continue
        if not _is_clean(rec):
            continue
        rows.append(rec)

    if not rows:
        raise SystemExit(
            f"no clean gos_original baselines in {jsonl_path}; "
            "did Phase A actually run with --bundle-types gos_original?"
        )

    # 同 query_id 可能有多行（重跑），保留最后一行（jsonl 顺序即时间序）。
    by_qid: dict[str, dict] = {}
    for r in rows:
        by_qid[r["query_id"]] = r
    rows = list(by_qid.values())

    pos = _pick(rows, target_reward=1.0, k=args.reward1_count)
    neg = _pick(rows, target_reward=0.0, k=args.reward0_count)
    picked = pos + neg

    if len(pos) < args.reward1_count:
        print(
            f"[warn] reward=1 候选不足: 需要 {args.reward1_count}，实得 {len(pos)}"
        )
    if len(neg) < args.reward0_count:
        print(
            f"[warn] reward=0 候选不足: 需要 {args.reward0_count}，实得 {len(neg)}"
        )

    out = [
        {
            "id": r["query_id"],
            "query": r["query"],
            "baseline_reward": float(r["reward"]),
            "baseline_runtime_sec": float(r.get("execution_time") or 0.0),
            "baseline_token_count": r.get("token_count"),
        }
        for r in picked
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"Selected {len(out)} baseline queries -> {out_path}")
    for r in out:
        print(
            f"  {r['id']:40s} reward={r['baseline_reward']} "
            f"runtime={r['baseline_runtime_sec']:.1f}s "
            f"tokens={r['baseline_token_count']}"
        )


if __name__ == "__main__":
    main()
