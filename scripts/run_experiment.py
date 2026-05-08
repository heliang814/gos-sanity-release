"""Run a sanity check: N queries x M bundle_types agent rollouts.

For each query:
  1. Retrieve GoS default bundle + full-library PPR scores.
  2. Build the requested perturbed bundles (delete_top, add_irrelevant, replace_similar).
  3. Run each requested bundle_type through harbor, append a JSONL line per run.

CLI 选项 ``--bundle-types`` 决定该次跑哪些 bundle_type（逗号分隔，默认 4 种全跑）。
两阶段流程下：Phase A 用 ``gos_original`` 跑 baseline，Phase C 用其余 3 种跑 perturbation。

Crash-safe: results/runs.jsonl is appended after every run. Re-running this
script with the same queries will produce duplicate JSONL lines -- analyze.py
groups by (query_id, bundle_type) so the latest still drives the verdict, but
clean state is wiping results/runs.jsonl beforehand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.agent_runner import RunRecord, _append_jsonl, run_once
from src.gos_interface import embed_query, load_library, retrieve
from src.perturbations import add_irrelevant, delete_top, replace_similar


ALL_BUNDLE_TYPES = ("gos_original", "delete_top", "add_irrelevant", "replace_similar")


def _record_skip(
    query_id: str,
    query: str,
    bundle_type: str,
    reason: str,
    jsonl_path: Path,
) -> None:
    """该 query 上某 perturbation 找不到合法候选时，写一行 skip 记录到 jsonl。

    skip 与 error 在 schema 上区分：error_type=None，skip_reason=非空。
    两阶段流程下只有 add_irrelevant 在没有 cosine_max 以下候选时才 skip；
    replace_similar 已改为永不 skip（除非非 bundle 池为空的病态情况）。
    """
    rec = RunRecord(
        query_id=query_id,
        query=query,
        bundle_type=bundle_type,
        skill_ids=[],
        skill_names=[],
        ppr_scores=[],
        token_count=None,
        agent_output="",
        reward=None,
        success=False,
        execution_time=0.0,
        error_type=None,
        skip_reason=reason,
    )
    _append_jsonl(rec, jsonl_path)
    print(f"[skip] {query_id} {bundle_type}: {reason} (recorded as skip row)")


def _parse_bundle_types(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--bundle-types is empty")
    bad = [p for p in parts if p not in ALL_BUNDLE_TYPES]
    if bad:
        raise ValueError(
            f"unknown bundle_type {bad!r}; valid choices: {ALL_BUNDLE_TYPES}"
        )
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--queries", default="data/queries.json")
    ap.add_argument(
        "--bundle-types",
        default=",".join(ALL_BUNDLE_TYPES),
        help=(
            "Comma-separated bundle_types to run. Valid: "
            f"{','.join(ALL_BUNDLE_TYPES)}. Default: all four."
        ),
    )
    args = ap.parse_args()

    requested = _parse_bundle_types(args.bundle_types)
    print(f"[run_experiment] bundle_types={requested}")

    cfg = yaml.safe_load(Path(args.config).read_text())
    queries = json.loads(Path(args.queries).read_text())
    library = load_library(cfg["paths"]["gos_repo"], cfg["paths"]["skills_library"])

    results_dir = Path(cfg["paths"]["results_dir"]).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "runs.jsonl"
    rng = np.random.default_rng(cfg["queries"]["selection_seed"])

    bundles_cache = results_dir / "default_bundles.json"
    cache: dict = json.loads(bundles_cache.read_text()) if bundles_cache.exists() else {}

    for q in tqdm(queries, desc="queries"):
        qid, query = q["id"], q["query"]

        # 1. Default bundle + full-library PPR (cache so reruns dont repay GoS)
        if qid in cache:
            bundle = cache[qid]["bundle"]
            ppr = cache[qid]["ppr"]
        else:
            bundle, ppr = retrieve(
                query,
                gos_repo=cfg["paths"]["gos_repo"],
                workspace=cfg["paths"]["gos_workspace"],
                top_n=cfg["retrieval"]["top_n"],
                max_context_chars=cfg["retrieval"]["max_context_chars"],
            )
            cache[qid] = {"bundle": bundle, "ppr": ppr}
            bundles_cache.write_text(json.dumps(cache, indent=2))

        # 2. Build per-bundle plan：(bundle_type, bundle, perturbation_meta)
        plan: list[tuple[str, list[str], dict]] = []

        if "gos_original" in requested:
            plan.append(("gos_original", bundle, {}))

        if "delete_top" in requested:
            res = delete_top(bundle, ppr, library=library)
            if res is None:
                _record_skip(qid, query, "delete_top", "empty_bundle", jsonl_path)
            else:
                plan.append(("delete_top", res["new_bundle"], res["meta"]))

        if "add_irrelevant" in requested:
            q_emb = embed_query(query, cfg["paths"]["gos_repo"])
            res = add_irrelevant(
                bundle, library, q_emb,
                cosine_max=cfg["perturbations"]["add_irrelevant_cosine_max"],
                rng=rng,
                ppr_scores=ppr,
            )
            if res is None:
                _record_skip(
                    qid, query, "add_irrelevant",
                    "no_candidate_below_cosine_max", jsonl_path,
                )
            else:
                plan.append(("add_irrelevant", res["new_bundle"], res["meta"]))

        if "replace_similar" in requested:
            res = replace_similar(
                bundle, ppr,
                library=library,
                rho_epsilon=cfg["perturbations"]["replace_similar_rho_epsilon"],
            )
            if res is None:
                # 病态情况：bundle 空 / 非 bundle 池空 / bundle 全无 PPR
                _record_skip(
                    qid, query, "replace_similar",
                    "pathological_no_swap_pair", jsonl_path,
                )
            else:
                plan.append(("replace_similar", res["new_bundle"], res["meta"]))

        # 3. Run each requested bundle_type
        for bundle_type, b, meta in plan:
            run_once(
                query_id=qid,
                query=query,
                bundle_type=bundle_type,
                bundle=b,
                library=library,
                ppr_scores=ppr,
                agent_cfg=cfg["agent"],
                paths_cfg=cfg["paths"],
                results_path=jsonl_path,
                perturbation_meta=meta,
            )

    print("\nDone. Inspect with: python scripts/analyze.py")


if __name__ == "__main__":
    main()
