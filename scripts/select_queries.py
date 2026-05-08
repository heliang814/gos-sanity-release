"""Pick 20 SkillsBench tasks for the sensitivity study.

Each task is a directory under evaluation/skillsbench/tasks/<name>/ with an
instruction file. Stratify by a coarse keyword bucket so the 20 queries span
multiple task types (file parsing, API calls, data, code, multi-skill etc.).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


_KEYWORD_BUCKETS = {
    "file_parse":   ("parse", "extract", "stl", "pdf", "csv", "json", "yaml", "xml"),
    "api_call":     ("api", "request", "endpoint", "http", "curl", "fetch"),
    "data_process": ("clean", "normalize", "aggregate", "transform", "join", "filter"),
    "code_run":     ("compile", "execute", "run", "test", "build", "script"),
    "multimedia":   ("video", "audio", "image", "frame", "scan"),
    "domain":       ("finance", "seismic", "macroeconomic", "power-grid"),
}


def _read_prompt(task_dir: Path) -> str:
    for fname in ("instruction.md", "INSTRUCTION.md", "README.md", "task.md"):
        f = task_dir / fname
        if f.exists():
            return f.read_text().strip()
    return task_dir.name


def _bucket(prompt: str) -> str:
    p = prompt.lower()
    for bucket, kws in _KEYWORD_BUCKETS.items():
        if any(kw in p for kw in kws):
            return bucket
    return "_other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="data/queries.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks_root = Path(cfg["paths"]["skillsbench_tasks"]).expanduser().resolve()
    if not tasks_root.exists():
        raise SystemExit(f"SkillsBench tasks not found: {tasks_root}")

    items: list[tuple[str, str, str]] = []
    for td in sorted(tasks_root.iterdir()):
        if not td.is_dir() or not (td / "environment").exists():
            continue
        prompt = _read_prompt(td)
        items.append((td.name, prompt, _bucket(prompt)))
    if not items:
        raise SystemExit(f"No valid tasks under {tasks_root}")

    rng = np.random.default_rng(cfg["queries"]["selection_seed"])
    by_bucket: dict[str, list] = defaultdict(list)
    for it in items:
        by_bucket[it[2]].append(it)
    for b in by_bucket.values():
        rng.shuffle(b)

    n = cfg["queries"]["count"]
    picked = []
    buckets = list(by_bucket.keys())
    rng.shuffle(buckets)
    while len(picked) < n:
        progress = False
        for b in buckets:
            if len(picked) >= n:
                break
            if by_bucket[b]:
                picked.append(by_bucket[b].pop())
                progress = True
        if not progress:
            break

    out = [
        {"id": tid, "query": prompt, "bucket": bucket}
        for tid, prompt, bucket in picked
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"Selected {len(out)} queries -> {args.out}")
    counts: dict[str, int] = defaultdict(int)
    for q in out:
        counts[q["bucket"]] += 1
    for b, c in sorted(counts.items()):
        print(f"  {b:14s} {c}")


if __name__ == "__main__":
    main()
