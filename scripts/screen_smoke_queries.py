"""扫 SkillsBench 任务 verifier，识别含"严格时间上界断言"的 risky task。

这类 task 的 verifier 含 ``assert elapsed < N (ms|s)`` 形式的边界性能断言。
agent 即使生成正确实现也很容易因 ~10% 的运行波动卡在阈值上，导致 baseline
本身 reward=0；任何 perturbation 也是 reward=0，**违反 HANDOFF.md 第 2 节**
"reward 在条件间有差异"判据，使 smoke test 失效。

设计目标
--------
* **不 hardcode task 名字**：基于 verifier 源码模式自动识别，新增 task 自动覆盖
* **结构化输出 + 证据**：写到 ``data/risky_tasks.json``，下游脚本可消费
* **保守匹配**：只命中明显的"严格时间上界"，不误伤反作弊下界
  （例如 ``assert elapsed >= 400`` 是"防止 mock 的下界"，不算 risky）

调用
----
``python scripts/screen_smoke_queries.py``
``setup_and_smoke.sh`` step 7 在 truncate query 前会先调用本脚本。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

# 时间相关变量名白名单（受限词表，避免命中无关变量）
_TIME_VAR = (
    r"(?:elapsed|duration|response_time|exec_time|render_time|load_time|"
    r"latency|wall_time|runtime|t_diff|delta_t|took)"
)

# 严格时间上界断言模式：
# 1. `assert <time_var> < <number>` （含 <=, 含小数）
_PAT_TIME_VAR_LT = re.compile(
    rf"assert\s+[^,#\n]*\b{_TIME_VAR}\b\s*<=?\s*[0-9]+(?:\.[0-9]+)?",
    re.IGNORECASE,
)
# 2. `assert (time.time() - t0) [* 1000] < N` 这种内联差值断言
_PAT_INLINE_TIMER_LT = re.compile(
    r"assert\s+\(?\s*time\.(?:time|perf_counter|monotonic)\(\)\s*-\s*\w+\s*\)?"
    r"\s*\*?\s*[0-9.]*\s*<=?\s*[0-9]+(?:\.[0-9]+)?",
)
_PATTERNS = [_PAT_TIME_VAR_LT, _PAT_INLINE_TIMER_LT]


def screen_task(task_dir: Path) -> list[dict]:
    """返回 task verifier 中所有命中"严格时间上界断言"的证据。

    扫 ``tests/`` 下所有 .py 文件。空列表表示 task 不含此类断言（视为 safe）。
    """
    tests_dir = task_dir / "tests"
    if not tests_dir.exists():
        return []

    evidence: list[dict] = []
    for f in sorted(tests_dir.rglob("*.py")):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            # 注释行跳过
            if stripped.startswith("#"):
                continue
            for pat in _PATTERNS:
                if pat.search(line):
                    evidence.append(
                        {
                            "file": str(f.relative_to(task_dir)),
                            "line": lineno,
                            "text": stripped[:200],
                        }
                    )
                    break
    return evidence


def main() -> None:
    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    tasks_root = Path(cfg["paths"]["skillsbench_tasks"]).expanduser().resolve()
    if not tasks_root.exists():
        raise SystemExit(f"SkillsBench tasks not found: {tasks_root}")

    n_total = 0
    risky: list[dict] = []
    for td in sorted(tasks_root.iterdir()):
        if not td.is_dir() or not (td / "environment").exists():
            continue
        n_total += 1
        ev = screen_task(td)
        if ev:
            risky.append({"task_id": td.name, "evidence": ev})

    out = Path("data/risky_tasks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(risky, indent=2, ensure_ascii=False))

    print(f"扫描 {n_total} task，命中 risky={len(risky)}")
    for r in risky:
        print(f"  ⚠ {r['task_id']}")
        for e in r["evidence"][:3]:
            print(f"      {e['file']}:{e['line']}  {e['text']}")
        if len(r["evidence"]) > 3:
            print(f"      ... 共 {len(r['evidence'])} 处命中")
    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
