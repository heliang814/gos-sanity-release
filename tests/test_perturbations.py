"""Offline 单元自检：合成 PPR 字典验证 perturbations 的 4 条契约。

不联网、不依赖 GoS / harbor / numpy 的 BLAS 资源。直接 ``python tests/test_perturbations.py``
即可运行。覆盖：
  1. 全部 PPR>0：取全局最近对（不一定是 bundle 内 PPR 最大的）
  2. 部分 bundle PPR=0：优先 PPR>0 的候选作为 swap-out
  3. 所有距离 > ε：返回 nearest_fallback，within_epsilon=False
  4. 所有距离 ≤ ε：返回 strict_within_epsilon，within_epsilon=True
  5. delete_top：删 bundle 中 PPR 最高的
  6. add_irrelevant：cosine_max 卡得严时返回 None；放宽时挑 cosine ≤ cosine_max 的一个

运行：
  cd gos-sanity && python tests/test_perturbations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 让 import src.perturbations 在不安装 package 的情况下工作
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.perturbations import (  # noqa: E402
    SkillRecord,
    add_irrelevant,
    delete_top,
    replace_similar,
)


def _mk_lib(items: list[tuple[str, str, np.ndarray, str]]) -> dict[str, SkillRecord]:
    return {
        sid: SkillRecord(skill_id=sid, name=name, embedding=emb, domain_tag=tag)
        for sid, name, emb, tag in items
    }


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------- delete_top ----------

def test_delete_top_picks_max_ppr() -> None:
    bundle = ["a", "b", "c"]
    ppr = {"a": 0.1, "b": 0.5, "c": 0.3}
    lib = _mk_lib([
        ("a", "Skill A", _norm(np.array([1.0, 0.0])), "x"),
        ("b", "Skill B", _norm(np.array([0.0, 1.0])), "y"),
        ("c", "Skill C", _norm(np.array([1.0, 1.0])), "z"),
    ])
    res = delete_top(bundle, ppr, library=lib)
    assert res is not None
    assert res["new_bundle"] == ["a", "c"]
    assert res["meta"]["deleted_skill_id"] == "b"
    assert res["meta"]["deleted_skill_name"] == "Skill B"
    assert abs(res["meta"]["deleted_skill_ppr"] - 0.5) < 1e-9
    print("OK  delete_top_picks_max_ppr")


def test_delete_top_empty_bundle_returns_none() -> None:
    assert delete_top([], {}, library={}) is None
    print("OK  delete_top_empty_bundle_returns_none")


# ---------- replace_similar ----------

def test_replace_similar_global_min_pair_all_pos() -> None:
    """全部 PPR>0，应取全局最近对（横扫所有 candidate × 所有非 bundle）。"""
    # bundle PPR: a=0.50, b=0.20, c=0.05
    # non-bundle PPR: x=0.49 (距 a 0.01), y=0.18 (距 b 0.02), z=0.03 (距 c 0.02)
    # 最小距离对：(a, x, 0.01)
    bundle = ["a", "b", "c"]
    ppr = {"a": 0.50, "b": 0.20, "c": 0.05, "x": 0.49, "y": 0.18, "z": 0.03}
    res = replace_similar(bundle, ppr, library=None, rho_epsilon=0.05)
    assert res is not None
    meta = res["meta"]
    assert meta["swapped_out_skill_id"] == "a", meta
    assert meta["swapped_in_skill_id"] == "x", meta
    assert abs(meta["rho_distance"] - 0.01) < 1e-9
    assert meta["within_epsilon"] is True
    assert meta["replace_mode"] == "strict_within_epsilon"
    assert res["new_bundle"] == ["x", "b", "c"]
    print("OK  replace_similar_global_min_pair_all_pos")


def test_replace_similar_prefer_pos_over_zero() -> None:
    """bundle 中既有 PPR>0 又有 PPR=0：优先 PPR>0 的作为 swap-out 候选。

    构造：a=0.30, b=0.0; non-bundle x=0.30000001 (距 a ~0)，y=0.0000001 (距 b ~0)。
    若不优先 PPR>0，b-y 距离更小，会换出 b；优先 PPR>0 时换出 a。
    """
    bundle = ["a", "b"]
    ppr = {"a": 0.30, "b": 0.0, "x": 0.30000001, "y": 0.0000001}
    res = replace_similar(bundle, ppr, library=None, rho_epsilon=0.05)
    assert res is not None
    meta = res["meta"]
    assert meta["swapped_out_skill_id"] == "a", meta
    assert meta["swapped_in_skill_id"] == "x", meta
    print("OK  replace_similar_prefer_pos_over_zero")


def test_replace_similar_nearest_fallback_when_no_eps() -> None:
    """所有候选对距离 > ε：返回 nearest_fallback，仍执行 swap。"""
    bundle = ["a", "b"]
    # a=0.50：nearest 非 bundle 为 x=0.30 (dist 0.20)；
    # b=0.20：nearest 非 bundle 为 x=0.30 (dist 0.10)。
    # 全局最小对 (b, x, 0.10)；ε=0.05 < 0.10 → fallback。
    ppr = {"a": 0.50, "b": 0.20, "x": 0.30, "y": 0.05}
    res = replace_similar(bundle, ppr, library=None, rho_epsilon=0.05)
    assert res is not None
    meta = res["meta"]
    assert meta["replace_mode"] == "nearest_fallback", meta
    assert meta["within_epsilon"] is False
    assert meta["swapped_out_skill_id"] == "b"
    assert meta["swapped_in_skill_id"] == "x"
    assert abs(meta["rho_distance"] - 0.10) < 1e-9
    print("OK  replace_similar_nearest_fallback_when_no_eps")


def test_replace_similar_strict_when_all_within_eps() -> None:
    """所有候选对距离 ≤ ε：strict_within_epsilon。"""
    bundle = ["a", "b"]
    ppr = {"a": 0.50, "b": 0.30, "x": 0.49, "y": 0.31}  # 0.01, 0.01
    res = replace_similar(bundle, ppr, library=None, rho_epsilon=0.05)
    assert res is not None
    meta = res["meta"]
    assert meta["replace_mode"] == "strict_within_epsilon"
    assert meta["within_epsilon"] is True
    print("OK  replace_similar_strict_when_all_within_eps")


def test_replace_similar_pathological_returns_none() -> None:
    """非 bundle 池为空：返回 None。"""
    bundle = ["a", "b"]
    ppr = {"a": 0.5, "b": 0.5}      # ppr_scores 中没有 bundle 之外的 skill
    assert replace_similar(bundle, ppr, library=None) is None
    # bundle 空也应该 None
    assert replace_similar([], {"x": 0.1}, library=None) is None
    print("OK  replace_similar_pathological_returns_none")


def test_replace_similar_all_bundle_zero_ppr_falls_back_to_full_pool() -> None:
    """bundle 全部 PPR=0：fallback 到全部 bundle 而非空候选集。"""
    bundle = ["a", "b"]
    ppr = {"a": 0.0, "b": 0.0, "x": 0.001, "y": 0.5}
    res = replace_similar(bundle, ppr, library=None, rho_epsilon=0.05)
    assert res is not None
    meta = res["meta"]
    # 最近邻：a->x (0.001), b->x (0.001)，按 b 字典序 tie-break，取 (a,x) 因为 a<b
    assert meta["swapped_in_skill_id"] == "x"
    assert meta["replace_mode"] == "strict_within_epsilon"
    print("OK  replace_similar_all_bundle_zero_ppr_falls_back_to_full_pool")


# ---------- add_irrelevant ----------

def test_add_irrelevant_picks_below_cosine_max() -> None:
    q = _norm(np.array([1.0, 0.0]))
    lib = _mk_lib([
        ("a", "A", _norm(np.array([1.0, 0.05])), "x"),     # cosine~1
        ("b", "B", _norm(np.array([-1.0, 0.0])), "y"),     # cosine~-1
        ("c", "C", _norm(np.array([0.0, 1.0])), "z"),      # cosine 0
    ])
    bundle = ["a"]
    rng = np.random.default_rng(42)
    res = add_irrelevant(bundle, lib, q, cosine_max=0.20, rng=rng, ppr_scores={"b": 0.0, "c": 0.0})
    assert res is not None
    meta = res["meta"]
    assert meta["added_skill_id"] in ("b", "c")
    assert meta["added_query_cosine"] <= 0.20 + 1e-9
    assert res["new_bundle"][-1] == meta["added_skill_id"]
    print("OK  add_irrelevant_picks_below_cosine_max")


def test_add_irrelevant_returns_none_when_no_candidate() -> None:
    q = _norm(np.array([1.0, 0.0]))
    lib = _mk_lib([
        ("a", "A", _norm(np.array([1.0, 0.05])), "x"),
        ("b", "B", _norm(np.array([1.0, 0.10])), "y"),
    ])
    res = add_irrelevant(["a"], lib, q, cosine_max=0.20)
    assert res is None
    print("OK  add_irrelevant_returns_none_when_no_candidate")


# ---------- main ----------

def main() -> None:
    test_delete_top_picks_max_ppr()
    test_delete_top_empty_bundle_returns_none()
    test_replace_similar_global_min_pair_all_pos()
    test_replace_similar_prefer_pos_over_zero()
    test_replace_similar_nearest_fallback_when_no_eps()
    test_replace_similar_strict_when_all_within_eps()
    test_replace_similar_pathological_returns_none()
    test_replace_similar_all_bundle_zero_ppr_falls_back_to_full_pool()
    test_add_irrelevant_picks_below_cosine_max()
    test_add_irrelevant_returns_none_when_no_candidate()
    print("\nALL PERTURBATION TESTS PASSED")


if __name__ == "__main__":
    main()
