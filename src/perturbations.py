"""Three perturbations of a GoS-retrieved skill bundle.

Each function returns a structured dict with:
  - ``new_bundle``: the perturbed list of skill_ids
  - ``meta``: perturbation-specific metadata to be persisted alongside the run

For ``add_irrelevant`` the function may still return ``None`` when no skill in
the library is below the cosine threshold (the design genuinely permits no
candidate); ``delete_top`` and ``replace_similar`` always return a result
unless the bundle / library is empty.

Caller pairs them with the original bundle to form the 4 conditions per query:
``gos_original``, ``delete_top``, ``add_irrelevant``, ``replace_similar``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    name: str
    embedding: np.ndarray            # unit-normalized
    domain_tag: str


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _name_of(library: dict[str, SkillRecord], skill_id: str) -> str:
    rec = library.get(skill_id) if library else None
    return rec.name if rec is not None else skill_id


def delete_top(
    bundle: list[str],
    ppr_scores: dict[str, float],
    library: dict[str, SkillRecord] | None = None,
) -> dict[str, Any] | None:
    """Drop the highest-PPR skill in the bundle.

    Hypothesis: if GoS's top pick is load-bearing, removing it should hurt
    reward. Invariant when the agent doesn't actually rely on the top skill.

    Returns ``None`` only if the bundle is empty. Otherwise returns:
      {
        "new_bundle": list[str],
        "meta": {"deleted_skill_id", "deleted_skill_name", "deleted_skill_ppr"},
      }
    """
    if not bundle:
        return None
    top = max(bundle, key=lambda s: ppr_scores.get(s, float("-inf")))
    new_bundle = [s for s in bundle if s != top]
    meta = {
        "deleted_skill_id": top,
        "deleted_skill_name": _name_of(library or {}, top),
        "deleted_skill_ppr": float(ppr_scores.get(top, 0.0)),
    }
    return {"new_bundle": new_bundle, "meta": meta}


def add_irrelevant(
    bundle: list[str],
    library: dict[str, SkillRecord],
    query_embedding: np.ndarray,
    cosine_max: float = 0.20,
    rng: np.random.Generator | None = None,
    ppr_scores: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Append one randomly-chosen skill that is semantically far from the query.

    Returns ``None`` when no library skill is below ``cosine_max``. Otherwise:
      {
        "new_bundle": list[str],
        "meta": {"added_skill_id", "added_skill_name", "added_skill_ppr",
                 "added_query_cosine"},
      }
    """
    rng = rng or np.random.default_rng()
    bundle_set = set(bundle)
    candidates: list[tuple[str, float]] = []
    for sid, rec in library.items():
        if sid in bundle_set:
            continue
        cs = _cosine(query_embedding, rec.embedding)
        if cs <= cosine_max:
            candidates.append((sid, cs))
    if not candidates:
        return None
    chosen_idx = int(rng.integers(0, len(candidates)))
    chosen_sid, chosen_cos = candidates[chosen_idx]
    new_bundle = list(bundle) + [chosen_sid]
    meta = {
        "added_skill_id": chosen_sid,
        "added_skill_name": _name_of(library, chosen_sid),
        "added_skill_ppr": float((ppr_scores or {}).get(chosen_sid, 0.0)),
        "added_query_cosine": float(chosen_cos),
    }
    return {"new_bundle": new_bundle, "meta": meta}


def replace_similar(
    bundle: list[str],
    ppr_scores: dict[str, float],
    library: dict[str, SkillRecord] | None = None,
    rho_epsilon: float = 0.05,
) -> dict[str, Any] | None:
    """Swap one bundle skill with its globally PPR-nearest non-bundle peer.

    Strategy
    --------
    1. Among bundle skills with a known PPR score, prefer those with PPR > 0
       as candidates to be swapped out (when at least one such exists).
    2. For each candidate, find its nearest non-bundle skill by absolute PPR
       difference. Compute (candidate, neighbor, distance) for all candidates.
    3. Pick the global pair with the smallest distance.
    4. If that distance ≤ ``rho_epsilon``, label as ``strict_within_epsilon``;
       otherwise label as ``nearest_fallback``. Either way perform the swap.

    The previous design picked one bundle skill at random and skipped when no
    epsilon-neighbor existed. On skills_200 the bundle tends to occupy local
    PPR maxima, so a random pick frequently lands on a skill with no neighbor
    inside ε and the perturbation gets dropped. The new design always returns
    a result unless the non-bundle pool is empty (a pathological case), and
    surfaces ``rho_distance`` + ``within_epsilon`` so downstream analysis can
    distinguish strict swaps from fallback swaps.

    Returns ``None`` only if either the bundle or the non-bundle pool with
    known PPR is empty. Otherwise:
      {
        "new_bundle": list[str],
        "meta": {
            "swapped_out_skill_id", "swapped_out_skill_name", "swapped_out_ppr",
            "swapped_in_skill_id",  "swapped_in_skill_name",  "swapped_in_ppr",
            "rho_distance", "within_epsilon", "replace_mode",
        },
      }
    """
    if not bundle:
        return None

    bundle_set = set(bundle)
    non_bundle_pairs = [
        (sid, float(rho)) for sid, rho in ppr_scores.items() if sid not in bundle_set
    ]
    if not non_bundle_pairs:
        return None

    # Bundle candidates that have a known PPR.
    bundle_with_rho = [
        (b, float(ppr_scores[b])) for b in bundle if b in ppr_scores
    ]
    if not bundle_with_rho:
        return None

    # Prefer PPR > 0 candidates.
    pos = [(b, r) for b, r in bundle_with_rho if r > 0.0]
    pool = pos if pos else bundle_with_rho

    # For each candidate, find its globally-nearest non-bundle neighbor.
    # Tie-break on neighbor id for determinism.
    best: tuple[str, float, str, float, float] | None = None  # (b, rb, n, rn, dist)
    for b, rb in pool:
        nearest_sid, nearest_rho = min(
            non_bundle_pairs,
            key=lambda nb: (abs(nb[1] - rb), nb[0]),
        )
        dist = abs(nearest_rho - rb)
        if best is None or (dist, b) < (best[4], best[0]):
            best = (b, rb, nearest_sid, nearest_rho, dist)

    assert best is not None  # guaranteed by the non-empty checks above
    swapped_out, rho_out, swapped_in, rho_in, dist = best
    within_eps = bool(dist <= rho_epsilon)
    mode = "strict_within_epsilon" if within_eps else "nearest_fallback"

    new_bundle = [swapped_in if s == swapped_out else s for s in bundle]
    meta = {
        "swapped_out_skill_id": swapped_out,
        "swapped_out_skill_name": _name_of(library or {}, swapped_out),
        "swapped_out_ppr": float(rho_out),
        "swapped_in_skill_id": swapped_in,
        "swapped_in_skill_name": _name_of(library or {}, swapped_in),
        "swapped_in_ppr": float(rho_in),
        "rho_distance": float(dist),
        "within_epsilon": within_eps,
        "replace_mode": mode,
    }
    return {"new_bundle": new_bundle, "meta": meta}
