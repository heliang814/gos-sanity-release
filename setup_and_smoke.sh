#!/usr/bin/env bash
#
# One-shot: set up env, clone GoS if needed, download data, install harbor,
# run a 4-rollout smoke test, then estimate the cost of the full 80-run sanity check.
#
# Usage:
#   bash setup_and_smoke.sh                # uses keys baked in below
#   bash setup_and_smoke.sh <ANTHROPIC_KEY> [<OPENAI_KEY>]   # override
#
# Layout assumed (auto-detected from script location):
#   <SK_EX_ROOT>/gos-sanity/setup_and_smoke.sh    (this file)
#   <SK_EX_ROOT>/graph-of-skills/                 (cloned by this script if missing)
#
# Re-runnable: skips downloads / installs that already exist.
#

set -euo pipefail

# --- key resolution: arg > env > .env.local ---
SCRIPT_DIR_FOR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR_FOR_ENV/.env.local"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
    echo "Loaded keys from $ENV_FILE"
fi

API_KEY="${1:-${ANTHROPIC_API_KEY:-}}"
OPENAI_KEY="${2:-${OPENAI_API_KEY:-}}"

if [ -z "$API_KEY" ] || [ -z "$OPENAI_KEY" ]; then
    cat >&2 <<EOF
ERROR: missing API keys. Three ways to provide them:

  1. As args (good for one-off):
     bash setup_and_smoke.sh '<anthropic_key>' '<openai_key>'

  2. As env vars (good for current shell):
     export ANTHROPIC_API_KEY=sk-ant-...
     export OPENAI_API_KEY=sk-...
     bash setup_and_smoke.sh

  3. As $ENV_FILE (one-time per machine; gitignored):
     cat > $ENV_FILE <<KEYS
     ANTHROPIC_API_KEY=sk-ant-...
     OPENAI_API_KEY=sk-...
     KEYS
     bash setup_and_smoke.sh
EOF
    exit 1
fi

if ! [[ "$API_KEY" =~ ^sk-ant- ]]; then
    echo "ERROR: ANTHROPIC key must start with 'sk-ant-'." >&2
    exit 1
fi
if ! [[ "$OPENAI_KEY" =~ ^sk- ]]; then
    echo "ERROR: OPENAI key must start with 'sk-'." >&2
    exit 1
fi

# --- paths (auto-detected) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANITY_DIR="$SCRIPT_DIR"
SK_EX_ROOT="$(dirname "$SCRIPT_DIR")"
GOS_DIR="$SK_EX_ROOT/graph-of-skills"

# --- OS detection ---
OS="$(uname -s)"
case "$OS" in
    Darwin) PROFILE="$HOME/.zshrc" ;;
    Linux)  PROFILE="$HOME/.bashrc" ;;
    *)      echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

step() { printf "\n========== STEP %s: %s ==========\n" "$1" "$2"; }
echo "OS=$OS  SK_EX_ROOT=$SK_EX_ROOT  PROFILE=$PROFILE"

# ============================================================
step 0 "Sanity-check Docker is installed and running"
# ============================================================
if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
ERROR: docker not found.

On macOS, install OrbStack (lighter than Docker Desktop):
  https://orbstack.dev
or Docker Desktop:
  https://www.docker.com/products/docker-desktop/

Start it, then re-run this script.
EOF
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker daemon not reachable. Start OrbStack/Docker Desktop and retry." >&2
    exit 1
fi
echo "docker: $(docker --version)"

# ============================================================
step 1 "Clone graph-of-skills if not present"
# ============================================================
if [ -d "$GOS_DIR/.git" ]; then
    echo "graph-of-skills already cloned at $GOS_DIR"
else
    echo "Cloning graph-of-skills via SSH..."
    git clone git@github.com:davidliuk/graph-of-skills.git "$GOS_DIR" || {
        echo "SSH clone failed; falling back to HTTPS via gh-proxy..."
        git clone https://gh-proxy.com/https://github.com/davidliuk/graph-of-skills.git "$GOS_DIR"
    }
fi

# ============================================================
step 2 "Set ANTHROPIC_API_KEY + OPENAI_API_KEY in this shell + persist"
# ============================================================
export ANTHROPIC_API_KEY="$API_KEY"
export ANTHROPIC_AUTH_TOKEN="$API_KEY"
export OPENAI_API_KEY="$OPENAI_KEY"
# 关键：清掉所有可能让容器内 claude 走代理 / 取错 token 的变量。
# 用户 ~/.bashrc 里有 ANTHROPIC_BASE_URL=https://api.gpugeek.com，
# 否则 harbor 会把它透传到容器，导致真 sk-ant 被发到 gpugeek 代理失败。
unset ANTHROPIC_BASE_URL ANTHROPIC_BASE_URLS ANTHROPIC_API_BASE \
      CLAUDE_CODE_OAUTH_TOKEN CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX \
      ANTHROPIC_BEDROCK_BASE_URL ANTHROPIC_VERTEX_PROJECT_ID

# 不再持久化到 PROFILE：用户的 ~/.claude/settings.json 已配置 gpugeek 代理，
# 把真 sk-ant key 写进 .bashrc 会和 settings.json 冲突，触发 claude
# "Auth conflict: both ANTHROPIC_AUTH_TOKEN and ANTHROPIC_API_KEY are set"。
# 脚本每次从 .env.local 读取并 export 到自己的子进程即可。
echo "Keys exported to current script process only (not persisted to $PROFILE)."

echo ">>> Verifying API key..."
HTTP_CODE=$(curl -s -o /tmp/anthropic_resp.json -w "%{http_code}" --max-time 20 \
    https://api.anthropic.com/v1/messages \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "content-type: application/json" \
    -d '{"model":"claude-sonnet-4-5","max_tokens":20,"messages":[{"role":"user","content":"say pong"}]}' \
    || echo "000")

echo "HTTP $HTTP_CODE"
echo "Response body:"
cat /tmp/anthropic_resp.json 2>/dev/null | python3 -m json.tool 2>/dev/null \
    || cat /tmp/anthropic_resp.json 2>/dev/null
echo

case "$HTTP_CODE" in
    200) echo ">>> API key OK" ;;
    401) echo "ERROR: 401 Unauthorized -- key is invalid." >&2; exit 1 ;;
    403) echo "ERROR: 403 Forbidden -- likely geo-blocked. Use a VPN with US/JP/EU exit." >&2; exit 1 ;;
    000) echo "ERROR: curl failed (timeout or DNS). Check network / VPN." >&2; exit 1 ;;
    *)   echo "ERROR: unexpected HTTP $HTTP_CODE." >&2; exit 1 ;;
esac

# ============================================================
step 3 "Download GoS minimal data (skills_200 + workspace + tasks)"
# ============================================================
cd "$GOS_DIR"
mkdir -p data/skillsets data/gos_workspace evaluation/skillsbench

# skills_200
if [ -d data/skillsets/skills_200 ] && [ "$(ls data/skillsets/skills_200 | wc -l | tr -d ' ')" -gt 100 ]; then
    echo "skills_200 already present, skipping"
else
    URL_HF="https://huggingface.co/datasets/DLPenn/graph-of-skills-data/resolve/main/skills_200.tar.gz"
    URL_MIRROR="https://hf-mirror.com/datasets/DLPenn/graph-of-skills-data/resolve/main/skills_200.tar.gz"
    echo ">>> Trying direct HuggingFace..."
    if ! curl -fL --retry 2 --max-time 600 -o /tmp/skills_200.tar.gz "$URL_HF" 2>/dev/null; then
        echo "Direct failed; trying hf-mirror..."
        curl -fL --retry 2 --max-time 600 -o /tmp/skills_200.tar.gz "$URL_MIRROR"
    fi
    [ -s /tmp/skills_200.tar.gz ] || { echo "ERROR: skills_200 download empty" >&2; exit 1; }
    ls -lh /tmp/skills_200.tar.gz
    tar xzf /tmp/skills_200.tar.gz -C data/skillsets/
fi
echo "skills_200 count: $(ls data/skillsets/skills_200 | wc -l | tr -d ' ')"

# workspace
if [ -d data/gos_workspace/skills_200_v1 ] && [ "$(ls data/gos_workspace/skills_200_v1 | wc -l | tr -d ' ')" -gt 0 ]; then
    echo "workspace already present, skipping"
else
    URL_HF="https://huggingface.co/datasets/DLPenn/graph-of-skills-data/resolve/main/gos_workspace_skills_200_v1.tar.gz"
    URL_MIRROR="https://hf-mirror.com/datasets/DLPenn/graph-of-skills-data/resolve/main/gos_workspace_skills_200_v1.tar.gz"
    echo ">>> Downloading workspace (try direct HF, fall back to hf-mirror)..."
    if ! curl -fL --retry 1 --connect-timeout 10 --max-time 900 -o /tmp/ws.tar.gz "$URL_HF"; then
        echo ">>> Direct HF failed/slow, switching to hf-mirror..."
        curl -fL --retry 2 --connect-timeout 10 --max-time 900 -o /tmp/ws.tar.gz "$URL_MIRROR"
    fi
    [ -s /tmp/ws.tar.gz ] || { echo "ERROR: workspace download empty" >&2; exit 1; }
    tar xzf /tmp/ws.tar.gz -C data/gos_workspace/
fi

# SkillsBench tasks
if [ -d evaluation/skillsbench/tasks ] && [ "$(ls evaluation/skillsbench/tasks | wc -l | tr -d ' ')" -gt 50 ]; then
    echo "SkillsBench tasks already present, skipping"
else
    URL_GH="https://github.com/benchflow-ai/skillsbench/archive/refs/heads/main.tar.gz"
    URL_MIRROR="https://gh-proxy.com/https://github.com/benchflow-ai/skillsbench/archive/refs/heads/main.tar.gz"
    echo ">>> Downloading SkillsBench tasks ~580MB (try direct GH, fall back to gh-proxy)..."
    if ! curl -fL --retry 1 --connect-timeout 10 --max-time 1800 -o /tmp/sb.tar.gz "$URL_GH"; then
        echo ">>> Direct GitHub failed/slow, switching to gh-proxy..."
        curl -fL --retry 2 --connect-timeout 10 --max-time 1800 -o /tmp/sb.tar.gz "$URL_MIRROR"
    fi
    [ -s /tmp/sb.tar.gz ] || { echo "ERROR: skillsbench download empty" >&2; exit 1; }
    tar xzf /tmp/sb.tar.gz -C /tmp/
    rm -f evaluation/skillsbench/tasks
    ln -sfn /tmp/skillsbench-main/tasks evaluation/skillsbench/tasks
fi
echo "task count: $(ls evaluation/skillsbench/tasks | wc -l | tr -d ' ')"

# ============================================================
step 4 "Install uv + GoS Python deps"
# ============================================================
cd "$GOS_DIR"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv sync

# ============================================================
step 5 "Install Harbor"
# ============================================================
if ! command -v harbor >/dev/null 2>&1; then
    uv tool install harbor
fi
export PATH="$HOME/.local/bin:$PATH"
harbor --version

# ============================================================
step 6 "Set up gos-sanity venv"
# ============================================================
cd "$SANITY_DIR"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .
if ! pip install --quiet -e "$GOS_DIR" 2>/tmp/gos_install.log; then
    echo "WARN: standard install failed; retrying with --ignore-requires-python"
    cat /tmp/gos_install.log | tail -5
    if ! pip install --quiet -e "$GOS_DIR" --ignore-requires-python 2>>/tmp/gos_install.log; then
        echo "WARN: still failing; installing core GoS deps directly"
        pip install --quiet loguru fast-graphrag pydantic litellm openai anthropic tiktoken
        export PYTHONPATH="$GOS_DIR:${PYTHONPATH:-}"
    fi
fi

# ============================================================
step 7 "Pick query + truncate to 1 for smoke test (skip risky tasks with strict perf-time asserts)"
# ============================================================
python scripts/select_queries.py
# 扫所有 task verifier，识别含"严格时间上界断言"的 risky task。
# 否则 baseline (gos_original) 自身 reward 即受边界性能波动卡住，
# perturbation 也是 reward=0，违反 HANDOFF.md 第 2 节"reward 在条件间有差异"判据。
python scripts/screen_smoke_queries.py
python3 - <<'PY'
import json, pathlib, sys
p = pathlib.Path('data/queries.json')
qs = json.loads(p.read_text())

risky_path = pathlib.Path('data/risky_tasks.json')
risky_ids = set()
if risky_path.exists():
    risky_ids = {r['task_id'] for r in json.loads(risky_path.read_text())}

# 从 select_queries 选出的 20 个里挑首个非 risky 的做 smoke。
chosen = next((q for q in qs if q['id'] not in risky_ids), None)
if chosen is None:
    sys.exit('ERROR: 全部候选 query 都被标为 risky，smoke 无法选 baseline-friendly 目标')

if chosen['id'] != qs[0]['id']:
    print(f"Note: 跳过 risky task '{qs[0]['id']}' (verifier 含严格时间上界断言, baseline 注定 reward=0)")
    print(f"      新 smoke target: '{chosen['id']}'")

p.write_text(json.dumps([chosen], indent=2, ensure_ascii=False))
print(f"Smoke test query: {chosen['id']}")
print(f"Prompt preview: {chosen['query'][:200]}")
PY

# ============================================================
step 8 "Run smoke test (4 rollouts)"
# ============================================================
# 真正的结果文件路径来自 experiment.yaml 的 paths.results_dir，
# 这里读出来再清空，避免脚本相对路径 "results/runs.jsonl" 与 yaml 配置不一致
# 导致清空/读取的不是同一个文件。
RESULTS_DIR=$(python3 -c "import yaml,os; p=yaml.safe_load(open('configs/experiment.yaml'))['paths']['results_dir']; print(os.path.expanduser(p))")
mkdir -p "$RESULTS_DIR"
> "$RESULTS_DIR/runs.jsonl"
echo "smoke test 输出 -> $RESULTS_DIR/runs.jsonl"
python scripts/run_experiment.py 2>&1 | tee /tmp/smoke.log

# ============================================================
step 9 "Token / cost extrapolation for full 80-run experiment"
# ============================================================
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY'
import json, os, pathlib
jsonl = pathlib.Path(os.environ['RESULTS_DIR']) / 'runs.jsonl'
runs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
print(f"\nRuns completed: {len(runs)}  (jsonl: {jsonl})")
for r in runs:
    sr = r.get('skip_reason')
    if sr:
        print(f"  {r['bundle_type']:20s} SKIPPED ({sr})")
    else:
        print(f"  {r['bundle_type']:20s} reward={r.get('reward')} tokens={r.get('token_count')} time={r['execution_time']:.1f}s err={r.get('error_type')}")

# 只用真正跑过的 trial（非 skip）做 token / cost 估算
ran = [r for r in runs if not r.get('skip_reason')]
toks = [r['token_count'] for r in ran if r.get('token_count')]
if toks:
    avg = sum(toks) / len(toks)
    # claude-sonnet-4-5 价格：$3/M input, $15/M output；近似按 95% input/5% output 拆
    cost_per_run = (avg * 0.95 * 3 + avg * 0.05 * 15) / 1_000_000
    full_cost = cost_per_run * 80
    print(f"\nAvg total tokens / run: {avg:.0f}")
    print(f"Estimated cost / run:    ${cost_per_run:.2f}")
    print(f"Extrapolate 80 runs:     ${full_cost:.0f}")
    print()
    if full_cost > 480:
        print("STOP: extrapolated > $480. Investigate before running full.")
    elif full_cost > 240:
        print("CAUTION: $240-480 range. Tighten timeout or accept.")
    else:
        print("OK: <= $240. Safe to top up balance and run full.")
else:
    print("WARN: no token_count in any run; check error_type and harbor logs.")

# HANDOFF.md 第 2 节的 4 条 smoke 通过判据自动校验（skip 不计入 error）
print("\n=== HANDOFF.md smoke 通过判据自检 ===")
not_null = [r for r in runs if r.get('reward') is not None]
rewards = [r['reward'] for r in not_null]
errs = [r.get('error_type') for r in runs if r.get('error_type')]
skips = [r for r in runs if r.get('skip_reason')]
c1 = len(not_null) > 0
c2 = len(set(rewards)) > 1 if rewards else False
c3 = (sum(toks)/len(toks) * (0.95*3 + 0.05*15) / 1_000_000 * 80) < 400 if toks else False
c4 = len(errs) <= 1
print(f"  #1 reward 不全 null:                       {'✅' if c1 else '❌'}")
print(f"  #2 reward 在条件间有差异（不全 0/全 1）:    {'✅' if c2 else '❌'}  实际 reward 集: {sorted(set(rewards))}")
print(f"  #3 80-run 外推 < $400:                     {'✅' if c3 else '❌'}")
print(f"  #4 error_type 至多 1 个非 None:            {'✅' if c4 else '❌'}  实际: {errs or '全 None'}")
if skips:
    print(f"  ℹ skip 行数: {len(skips)}（perturbation 找不到合法候选，不算 error，已写入 jsonl 保留 row）")
print()
if all([c1, c2, c3, c4]):
    print("✅ 全部 4 条通过 — 可以跑全量 80：python scripts/select_queries.py && python scripts/run_experiment.py")
else:
    print("❌ 至少一条不过 — 不要跑全量；先排查上面打 ❌ 的判据。")
PY

# ============================================================
step 10 "Render bar chart (HANDOFF 文档 §2 提到的 bundle_vs_reward.png)"
# ============================================================
python scripts/analyze.py 2>&1 | tee -a /tmp/smoke.log

echo ""
echo "Smoke test done."
echo "  Results:    $RESULTS_DIR/runs.jsonl"
echo "  Bar chart:  $RESULTS_DIR/bundle_vs_reward.png"
