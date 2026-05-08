"""Run one (query, bundle) trial via Harbor and persist a JSONL line.

The bundle is materialized by symlinking each chosen skill package into a
temp directory, then setting SKILLSBENCH_SKILLS_HOST_DIR so the SkillsBench
docker-compose mounts it at /opt/skillsbench/skills. The agent therefore sees
exactly the bundle we picked -- no GoS runtime retrieval, no full library.

Output schema is fixed by the experiment design:
  query_id, query, bundle_type, skill_ids, skill_names, ppr_scores,
  token_count, agent_output, reward, success, execution_time, error_type
One JSON object per line, appended to results/runs.jsonl.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    query_id: str
    query: str
    bundle_type: str                 # "gos_original" | "delete_top" | "add_irrelevant" | "replace_similar"
    skill_ids: list[str]
    skill_names: list[str]
    ppr_scores: list[float]
    token_count: int | None
    agent_output: str
    reward: float | None
    success: bool
    execution_time: float
    error_type: str | None
    # 该 trial 因 perturbation 不适用而未实际跑 agent 时的原因（如
    # "no_ppr_neighbor_within_epsilon"）；正常 trial 为 None。
    # 与 error_type 的语义区别：error_type 是 agent/verifier/基建出错；
    # skip_reason 是设计上 perturbation 找不到合法候选——不算"错误"，
    # 但需写入 jsonl 让 80 行结果完整可分析。
    skip_reason: str | None = None
    # Perturbation-specific metadata (delete_top / add_irrelevant / replace_similar)。
    # gos_original 时为空 dict。schema 由 src/perturbations.py 决定。
    perturbation_meta: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def _stage_bundle(bundle: list[str], full_library_dir: Path) -> Path:
    """Build a temp dir containing only the bundle's skill packages (symlinks)."""
    stage = Path(tempfile.mkdtemp(prefix="gos_sanity_"))
    for skill_id in bundle:
        src = full_library_dir / skill_id
        if not src.exists():
            shutil.rmtree(stage, ignore_errors=True)
            raise FileNotFoundError(
                f"Skill {skill_id!r} not found in {full_library_dir}"
            )
        (stage / skill_id).symlink_to(src.resolve())
    return stage


def _build_extra_mounts() -> list[dict]:
    """构造容器额外 bind 挂载列表。

    目前只做一件事：把 host 上的 playwright chromium 缓存挂到容器，
    避免 react-performance-debugging / fix-visual-stability 两个 task 的
    test.sh 在容器内重新从 playwright.azureedge.net 下载 chromium 失败。
    要求 host 上先 `python3 -m playwright install chromium`（>= 1.49.1）。
    """
    mounts: list[dict] = []
    pw_cache = Path.home() / ".cache" / "ms-playwright"
    if pw_cache.exists():
        # 不能 read_only：playwright install 会写 __dirlock，即便 chromium 已存在。
        # gos-sanity trial 串行执行，不会有并发写冲突。
        mounts.append({
            "type": "bind",
            "source": str(pw_cache),
            "target": "/root/.cache/ms-playwright",
        })
    return mounts


def _harbor_run(
    task_dir: Path,
    out_dir: Path,
    agent_cfg: dict,
    bundle_skills_dir: Path,
    timeout_s: int,
) -> subprocess.CompletedProcess:
    backend_to_agent = {"openai": "codex", "anthropic": "claude-code", "gemini": "gemini-cli"}
    cmd = [
        "harbor", "run",
        "--agent", backend_to_agent.get(agent_cfg["backend"], agent_cfg["backend"]),
        "--model", agent_cfg["model"],
        "--force-build",
        "--timeout-multiplier", str(agent_cfg.get("harbor_timeout_multiplier", 5)),
        "-p", str(task_dir),
        "-o", str(out_dir),
    ]
    extra_mounts = _build_extra_mounts()
    if extra_mounts:
        cmd += ["--mounts-json", json.dumps(extra_mounts)]
    env = {**os.environ, "SKILLSBENCH_SKILLS_HOST_DIR": str(bundle_skills_dir)}
    # 强制走 Anthropic 官方 API。
    # 主机 ~/.claude/settings.json 里有 ANTHROPIC_API_KEY=d0ziw...（gpugeek token）和
    # ANTHROPIC_BASE_URL=https://api.gpugeek.com，harbor 会把它们透传到容器。
    # 必须用 .env.local 里的真 sk-ant- key 覆盖，并 unset BASE_URL，否则容器内
    # claude-code 会报 billing_error（gpugeek 余额不足）或 api_retry 死循环。
    _env_local = Path(__file__).parent.parent / ".env.local"
    _real_key = None
    if _env_local.exists():
        for _line in _env_local.read_text().splitlines():
            if _line.startswith("ANTHROPIC_API_KEY="):
                _real_key = _line.split("=", 1)[1].strip()
                break
    if _real_key and _real_key.startswith("sk-ant-"):
        env["ANTHROPIC_API_KEY"] = _real_key
    for k in (
        "ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URLS", "ANTHROPIC_API_BASE",
        "ANTHROPIC_BEDROCK_BASE_URL", "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        env.pop(k, None)
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False
    )


def _read_harbor_result(out_dir: Path) -> dict:
    """读取 harbor 写出的 trial result.json。

    harbor 在 out_dir 下会同时写两个 result.json：
      out_dir/<ts>/result.json                  ← batch summary（含 stats，无 reward）
      out_dir/<ts>/<trial_name>/result.json     ← 单个 trial 结果（含 verifier_result/agent_result）
    我们要的是后者。优先选含 ``verifier_result`` 字段的那个，避免任何路径深度
    断言耦合到 harbor 内部目录约定。
    """
    candidates = list(out_dir.glob("**/result.json"))
    if not candidates:
        return {}
    for c in candidates:
        try:
            data = json.loads(c.read_text())
        except json.JSONDecodeError:
            continue
        if "verifier_result" in data or "agent_result" in data:
            return data
    return json.loads(candidates[0].read_text())


def _classify_error(payload: dict, harbor_stderr: str) -> str | None:
    """Coarse error bucket. None when the run completed and verifier returned a reward.

    优先级：exception_info > agent_tokens==0 > reward 存在性。
    verifier 在 agent setup 失败时仍会写 reward=0.0（看到空 solution），
    必须先排除 exception 才能信任 reward 值。
    """
    # 1. exception_info 优先——agent/harbor 内部异常
    exc_type = (payload.get("exception_info") or {}).get("exception_type") or ""
    if exc_type:
        if "Timeout" in exc_type or "timeout" in exc_type.lower():
            return "agent_timeout"
        if "NonZeroAgentExitCodeError" in exc_type:
            # agent 进程非零退出：可能是 install 失败或 claude-code 本身 crash
            ar = payload.get("agent_result") or {}
            tok = (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0)
            return "agent_setup_failed" if tok == 0 else "agent_failed"
        return f"harbor_exception:{exc_type}"

    # 2. harbor_stderr 兜底（exception_info 缺失时）
    if "Timeout" in harbor_stderr or "timeout" in harbor_stderr.lower():
        return "agent_timeout"
    if "NonZeroAgentExitCodeError" in harbor_stderr:
        return "agent_setup_failed"

    # 3. 无 exception：看 reward
    if not payload:
        return "harbor_no_result"
    vr = payload.get("verifier_result") or {}
    rewards = vr.get("rewards") or {}
    if rewards.get("reward") is not None:
        return None  # 正常完成
    if payload.get("agent_result") is None:
        return "agent_failed"
    if payload.get("verifier_result") is None:
        return "verifier_failed"
    return "unknown"


def run_once(
    *,
    query_id: str,
    query: str,
    bundle_type: str,
    bundle: list[str],
    library: dict,                       # skill_id -> SkillRecord (for skill_names)
    ppr_scores: dict[str, float],
    agent_cfg: dict,
    paths_cfg: dict,
    results_path: Path,
    perturbation_meta: dict[str, Any] | None = None,
) -> RunRecord:
    skill_names = [getattr(library.get(s), "name", s) for s in bundle]
    bundle_ppr = [float(ppr_scores.get(s, 0.0)) for s in bundle]

    record = RunRecord(
        query_id=query_id,
        query=query,
        bundle_type=bundle_type,
        skill_ids=list(bundle),
        skill_names=skill_names,
        ppr_scores=bundle_ppr,
        token_count=None,
        agent_output="",
        reward=None,
        success=False,
        execution_time=0.0,
        error_type=None,
        perturbation_meta=dict(perturbation_meta or {}),
    )

    t0 = time.time()
    stage_dir: Path | None = None
    try:
        stage_dir = _stage_bundle(
            bundle, Path(paths_cfg["skills_library"]).expanduser().resolve()
        )
        task_dir = (
            Path(paths_cfg["skillsbench_tasks"]).expanduser().resolve() / query_id
        )
        if not task_dir.exists():
            raise FileNotFoundError(f"SkillsBench task missing: {task_dir}")

        out_dir = Path(paths_cfg["results_dir"]) / "harbor_jobs" / f"{query_id}__{bundle_type}__{record.run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        proc = _harbor_run(
            task_dir, out_dir, agent_cfg, stage_dir,
            timeout_s=int(agent_cfg.get("timeout_s", 1800)),
        )
        payload = _read_harbor_result(out_dir) or {}
        # 防御：harbor 写出 "verifier_result": null / "agent_result": null（agent 容器 setup 失败常见）
        rewards = (payload.get("verifier_result") or {}).get("rewards") or {}
        agent_result = payload.get("agent_result") or {}

        record.reward = (
            float(rewards["reward"]) if rewards.get("reward") is not None else None
        )
        record.success = bool(record.reward and record.reward > 0)
        record.token_count = (
            (agent_result.get("n_input_tokens") or 0) + (agent_result.get("n_output_tokens") or 0)
        ) or None
        record.agent_output = (agent_result.get("final_output") or "")[:8000]
        record.error_type = _classify_error(payload, proc.stderr)

    except subprocess.TimeoutExpired:
        record.error_type = "agent_timeout"
    except FileNotFoundError as exc:
        record.error_type = f"missing:{exc}"
    except Exception as exc:                           # noqa: BLE001
        record.error_type = f"exception:{type(exc).__name__}:{exc}"
    finally:
        record.execution_time = time.time() - t0
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)
        _append_jsonl(record, results_path)

    return record


def _append_jsonl(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
