"""Aggregate runs.jsonl: mean reward by bundle_type, plus a Case A / Case B verdict.

Case A: perturbations move reward in interpretable directions
        -> bundle composition matters, surrogate has signal to learn.
Case B: large noise within the same (query, bundle_type) or no consistent
        gap between gos_original and perturbations -> too noisy to conclude.

Also dumps a matplotlib bar chart of mean reward by bundle_type.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import yaml


BUNDLE_ORDER = ("gos_original", "delete_top", "add_irrelevant", "replace_similar")


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"runs.jsonl not found at {path}. Run scripts/run_experiment.py first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aggregate(runs: list[dict]) -> dict:
    """Group rewards by bundle_type. Per-query rewards are also returned so the
    caller can compute paired deltas (within-query, between bundle types).
    """
    by_type: dict[str, list[float]] = defaultdict(list)
    by_query_type: dict[tuple[str, str], list[float]] = defaultdict(list)
    failures: dict[str, int] = defaultdict(int)

    for r in runs:
        bt = r["bundle_type"]
        if r.get("reward") is None:
            failures[bt] += 1
            continue
        by_type[bt].append(float(r["reward"]))
        by_query_type[(r["query_id"], bt)].append(float(r["reward"]))

    return {"by_type": by_type, "by_query_type": by_query_type, "failures": failures}


def paired_deltas(by_query_type: dict, baseline: str = "gos_original") -> dict[str, list[float]]:
    """For each perturbed bundle_type, compute per-query (R_perturbed - R_baseline)."""
    out: dict[str, list[float]] = defaultdict(list)
    queries = {qid for qid, bt in by_query_type if bt == baseline}
    for qid in queries:
        base = by_query_type.get((qid, baseline), [])
        if not base:
            continue
        b = mean(base)
        for bt in BUNDLE_ORDER:
            if bt == baseline:
                continue
            pert = by_query_type.get((qid, bt), [])
            if not pert:
                continue
            out[bt].append(mean(pert) - b)
    return out


def render_plot(by_type: dict[str, list[float]], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping plot")
        return

    types = [t for t in BUNDLE_ORDER if t in by_type]
    means = [mean(by_type[t]) for t in types]
    errs = [pstdev(by_type[t]) if len(by_type[t]) >= 2 else 0.0 for t in types]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = list(range(len(types)))
    ax.bar(x, means, yerr=errs, capsize=5,
           color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=15, ha="right")
    ax.set_ylabel("mean reward")
    ax.set_title("Bundle perturbation vs reward")
    ax.set_ylim(0, max(1.0, max(means) + max(errs) + 0.1))
    ax.axhline(means[0] if means else 0, color="grey", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--noise-warn-threshold", type=float, default=0.30,
                    help="If pstdev of within-query reward >= this, flag Case B noise.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results_dir = Path(cfg["paths"]["results_dir"]).expanduser().resolve()
    runs = _load(results_dir / "runs.jsonl")
    agg = aggregate(runs)

    print(f"\nLoaded {len(runs)} runs.\n")
    print(f"{'bundle_type':<20} {'n':>4} {'mean':>7} {'std':>7} {'failures':>9}")
    print("-" * 55)
    for bt in BUNDLE_ORDER:
        rs = agg["by_type"].get(bt, [])
        n = len(rs)
        m = mean(rs) if rs else float("nan")
        s = pstdev(rs) if len(rs) >= 2 else 0.0
        f = agg["failures"].get(bt, 0)
        print(f"{bt:<20} {n:>4} {m:>7.3f} {s:>7.3f} {f:>9}")

    deltas = paired_deltas(agg["by_query_type"])
    if deltas:
        print(f"\nPaired delta vs gos_original (per-query mean of perturbed - baseline):")
        for bt in BUNDLE_ORDER:
            if bt == "gos_original" or bt not in deltas:
                continue
            d = deltas[bt]
            print(f"  {bt:<20} mean_delta={mean(d):+.3f}  n={len(d)}  "
                  f"std={pstdev(d) if len(d) >= 2 else 0:.3f}")

    # Case A vs Case B verdict (heuristic; calibrate after first run)
    base = agg["by_type"].get("gos_original", [])
    base_mean = mean(base) if base else 0.0
    moves = [abs(mean(d)) for d in deltas.values()] if deltas else []
    biggest_move = max(moves) if moves else 0.0

    verdict_lines = []
    if not base:
        verdict_lines.append("No gos_original runs succeeded -- cannot judge.")
    elif biggest_move >= 0.10:
        verdict_lines.append(
            f"Case A (likely): largest |paired delta| = {biggest_move:.2f}, "
            f"baseline mean = {base_mean:.2f}. Bundle composition appears to move reward."
        )
    else:
        verdict_lines.append(
            f"Case B (likely): largest |paired delta| = {biggest_move:.2f}, "
            f"baseline mean = {base_mean:.2f}. Either too noisy or perturbations dont matter; "
            f"add repeats for high-variance queries before drawing a conclusion."
        )

    print("\nVerdict:")
    for line in verdict_lines:
        print("  " + line)

    render_plot(agg["by_type"], results_dir / "bundle_vs_reward.png")


if __name__ == "__main__":
    main()
