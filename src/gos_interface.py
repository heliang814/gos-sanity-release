"""Wrapper around the GoS retrieval pipeline at github.com/davidliuk/graph-of-skills.

Two calls expose what the rest of the project needs:
    load_library(gos_repo, library_path) -> {skill_id: SkillRecord}
    retrieve(query, ...)                 -> (bundle, ppr_scores) where ppr is full-library

This module is the only place that imports gos.* — keep it that way so the
rest of the code stays in plain Python types.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

from .perturbations import SkillRecord


_FULL_TOPN = 10_000
_FULL_CTX_CHARS = 10**9


def _ensure_gos_on_path(gos_repo: str | Path) -> None:
    p = Path(gos_repo).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"GoS repo not found at {p}. "
            "git clone https://github.com/davidliuk/graph-of-skills"
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


_PERSISTENT_LOOP: asyncio.AbstractEventLoop | None = None


def _run(coro):
    """Run an async coro from sync code, reusing one event loop across calls.

    asyncio.run() creates a new loop each call -- litellm's background
    LoggingWorker binds to the first loop and then complains 'bound to a
    different event loop' on later calls. Reusing a single loop fixes that.
    """
    global _PERSISTENT_LOOP
    if _PERSISTENT_LOOP is None or _PERSISTENT_LOOP.is_closed():
        _PERSISTENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_PERSISTENT_LOOP)
    if _PERSISTENT_LOOP.is_running():
        return asyncio.run_coroutine_threadsafe(coro, _PERSISTENT_LOOP).result()
    return _PERSISTENT_LOOP.run_until_complete(coro)


def _build_rag(gos_repo: str | Path, workspace: str | Path):
    _ensure_gos_on_path(gos_repo)
    from gos import SkillGraphRAG                                         # type: ignore
    from gos.core.engine import (                                         # type: ignore
        build_default_embedding_service,
        build_default_llm_service,
    )

    workspace = str(Path(workspace).expanduser().resolve())
    return SkillGraphRAG(
        working_dir=workspace,
        config=SkillGraphRAG.Config(
            working_dir=workspace,
            prebuilt_working_dir=workspace,
            llm_service=build_default_llm_service(),
            embedding_service=build_default_embedding_service(),
            enable_query_rewrite=False,
        ),
    )


def load_library(gos_repo: str | Path, library_path: str | Path) -> dict[str, SkillRecord]:
    """Parse skill packages on disk into SkillRecords. Embedding model matches GoS."""
    _ensure_gos_on_path(gos_repo)
    from gos.core.engine import build_default_embedding_service           # type: ignore
    from gos.core.parsing import parse_skill_document                     # type: ignore

    lib = Path(library_path).expanduser().resolve()
    if not lib.exists():
        raise FileNotFoundError(
            f"Skill library not found: {lib}. "
            "Download via GoS scripts/download_data.sh --skillsets"
        )

    embedder = build_default_embedding_service()
    pending: list[tuple[str, str, str]] = []   # (skill_id, name, primary_tag)
    texts: list[str] = []

    for sd in sorted(p for p in lib.iterdir() if p.is_dir()):
        spec = sd / "SKILL.md"
        if not spec.exists():
            continue
        parsed = parse_skill_document(spec.read_text())
        if parsed is None:
            continue
        name = getattr(parsed, "name", None) or sd.name
        # ParsedSkillDocument.domain_tags is list[str]
        tags = getattr(parsed, "domain_tags", None) or []
        primary_tag = (tags[0].strip() if tags else "") or "_unknown"
        text = ((getattr(parsed, "one_line_capability", "") or "")
                + "\n" + (getattr(parsed, "description", "") or "")).strip() or name
        pending.append((sd.name, name, primary_tag))
        texts.append(text)

    vecs = np.asarray(_run(embedder.encode(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    return {
        sid: SkillRecord(skill_id=sid, name=name, embedding=v, domain_tag=tag)
        for (sid, name, tag), v in zip(pending, vecs)
    }


def _skill_id_of(retrieved_skill) -> str:
    """RetrievedSkill 对象的 skill_id（=skills_200 目录名）来源。

    GoS 的 ``RetrievedSkill.name`` 字段对部分 skill 返回 slug（如 ``xlsx``），
    对另一部分返回 human-readable 显示名（如 ``Automatic Speech Recognition (ASR)``）——
    取决于该 skill 的 SKILL.md 在 ``name:`` 字段写了什么。

    下游 ``_stage_bundle`` 需要的是目录名（=skill_id），统一用 ``source_path``
    的父目录来恢复，对所有 skill 一致。
    """
    sp = getattr(retrieved_skill, "source_path", None)
    if not sp:
        raise RuntimeError(
            f"RetrievedSkill 缺少 source_path 字段，无法恢复 skill_id: "
            f"name={getattr(retrieved_skill, 'name', '?')!r}"
        )
    return Path(sp).parent.name


def retrieve(
    query: str,
    gos_repo: str | Path,
    workspace: str | Path,
    top_n: int = 8,
    max_context_chars: int = 32_000,
) -> tuple[list[str], dict[str, float]]:
    """Run GoS twice: production budget for the bundle, unbounded for full-library
    PPR scores (needed by replace_similar to find PPR-neighbor candidates).
    """
    rag = _build_rag(gos_repo, workspace)
    bundle_res = _run(rag.async_retrieve(
        query, top_n=top_n, max_context_chars=max_context_chars,
    ))
    bundle = [_skill_id_of(s) for s in bundle_res.skills]

    full_res = _run(rag.async_retrieve(
        query, top_n=_FULL_TOPN, max_context_chars=_FULL_CTX_CHARS,
    ))
    ppr = {_skill_id_of(s): float(s.score) for s in full_res.skills}

    missing = set(bundle) - set(ppr)
    if missing:
        raise RuntimeError(
            f"PPR coverage gap: {len(missing)} skills in default bundle missing from full retrieval. "
            "Bump _FULL_TOPN or _FULL_CTX_CHARS."
        )

    return bundle, ppr


def embed_query(text: str, gos_repo: str | Path) -> np.ndarray:
    """Unit-normalized query embedding using GoS's embedding service."""
    _ensure_gos_on_path(gos_repo)
    from gos.core.engine import build_default_embedding_service           # type: ignore

    embedder = build_default_embedding_service()
    v = np.asarray(_run(embedder.encode([text])), dtype=np.float32)[0]
    n = float(np.linalg.norm(v))
    return v if n == 0 else v / n
