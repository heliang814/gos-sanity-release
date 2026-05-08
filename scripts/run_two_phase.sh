#!/usr/bin/env bash
#
# 5×4 两阶段 sanity check：setup（与 setup_and_smoke.sh 一致）+ 两阶段 run
#
# Phase A: $BASELINE_COUNT query × gos_original          → baseline
# Phase B: 从 baseline 里筛出 ${POS_COUNT}+${NEG_COUNT} 个目标 query
# Phase C: 5 query × {delete_top, add_irrelevant, replace_similar}
# Phase D: analyze.py 出图
#
# 用法（与 setup_and_smoke.sh 完全一致的 key 三路解析）：
#   bash scripts/run_two_phase.sh                         # uses keys from .env.local / env
#   bash scripts/run_two_phase.sh <ANTHROPIC_KEY> [<OPENAI_KEY>]
#   bash scripts/run_two_phase.sh --baseline-count 4 --select-pos 1 --select-neg 1   # smoke 量级
#
# 可重跑：已下载/已安装的步骤会跳过。jsonl 在 Phase A 开始前重置。
#

set -euo pipefail

# --- 解析 CLI flags（保持位置参数 = key 的语义） ---
BASELINE_COUNT=15
POS_COUNT=3
NEG_COUNT=2
CONFIG="configs/experiment.yaml"

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --baseline-count) BASELINE_COUNT="$2"; shift 2;;
        --select-pos)     POS_COUNT="$2"; shift 2;;
        --select-neg)     NEG_COUNT="$2"; shift 2;;
        --config)         CONFIG="$2"; shift 2;;
        --) shift; while [ $# -gt 0 ]; do POSITIONAL+=("$1"); shift; done;;
        -*) echo "unknown flag: $1" >&2; exit 1;;
        *)  POSITIONAL+=("$1"); shift;;
    esac
done
set -- "${POSITIONAL[@]:-}"

# --- key resolution: arg > env > .env.local ---
SCRIPT_DIR_FOR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANITY_DIR_FOR_ENV="$(dirname "$SCRIPT_DIR_FOR_ENV")"
ENV_FILE="$SANITY_DIR_FOR_ENV/.env.local"
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
     bash scripts/run_two_phase.sh '<anthropic_key>' '<openai_key>'

  2. As env vars (good for current shell):
     export ANTHROPIC_API_KEY=sk-ant-...
     export OPENAI_API_KEY=sk-...
     bash scripts/run_two_phase.sh

  3. As $ENV_FILE (one-time per machine; gitignored):
     cat > $ENV_FILE <<KEYS
     ANTHROPIC_API_KEY=sk-ant-...
     OPENAI_API_KEY=sk-...
     KEYS
     bash scripts/run_two_phase.sh
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
SANITY_DIR="$(dirname "$SCRIPT_DIR")"
SK_EX_ROOT="$(dirname "$SANITY_DIR")"
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
echo "BASELINE_COUNT=$BASELINE_COUNT  POS_COUNT=$POS_COUNT  NEG_COUNT=$NEG_COUNT  CONFIG=$CONFIG"

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
export OPENAI_API_KEY="$OPENAI_KEY"

# 注意：不 export ANTHROPIC_AUTH_TOKEN。claude-code v2.x 在同时看到
# ANTHROPIC_API_KEY + ANTHROPIC_AUTH_TOKEN 时会发"Auth conflict"警告。
# sk-ant- 开头的是 API key（走 x-api-key header），用 ANTHROPIC_API_KEY 即可。
# 历史上 setup_and_smoke.sh 同时 export 两个是为兼容老 SDK，这里去掉。

if ! grep -q "ANTHROPIC_API_KEY=" "$PROFILE" 2>/dev/null; then
    cat >> "$PROFILE" <<EOF
export ANTHROPIC_API_KEY='$API_KEY'
export OPENAI_API_KEY='$OPENAI_KEY'
EOF
    echo "Added API keys to $PROFILE"
else
    echo "$PROFILE already has ANTHROPIC_API_KEY (not overwriting)"
fi

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
# shellcheck disable=SC1091
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

# --- resolve results path from yaml（与原 step 8 一致）---
RESULTS_DIR=$(python3 -c "import yaml,os; p=yaml.safe_load(open('$CONFIG'))['paths']['results_dir']; print(os.path.expanduser(p))")
mkdir -p "$RESULTS_DIR"
JSONL="$RESULTS_DIR/runs.jsonl"

# ============================================================
step 7 "Phase A: select queries → take first $BASELINE_COUNT → run gos_original baseline"
# ============================================================
python scripts/select_queries.py --config "$CONFIG" --out data/queries.json
python3 - <<PY
import json, pathlib
p = pathlib.Path('data/queries.json')
qs = json.loads(p.read_text())
n = $BASELINE_COUNT
qs = qs[:n]
out = pathlib.Path('data/queries_baseline.json')
out.write_text(json.dumps(qs, indent=2, ensure_ascii=False))
print(f"Phase A baseline queries written: {out} ({len(qs)} queries)")
PY

# 清空 jsonl，避免与之前的 smoke 行混淆（与原 setup_and_smoke step 8 同样做法）
> "$JSONL"
echo "Phase A jsonl reset: $JSONL"

python scripts/run_experiment.py \
    --config "$CONFIG" \
    --queries data/queries_baseline.json \
    --bundle-types gos_original 2>&1 | tee /tmp/two_phase_A.log

# ============================================================
step 8 "Phase B: select ${POS_COUNT}×reward=1 + ${NEG_COUNT}×reward=0 baseline queries"
# ============================================================
python scripts/select_baseline_queries.py \
    --config "$CONFIG" \
    --out data/queries_selected.json \
    --reward1-count "$POS_COUNT" \
    --reward0-count "$NEG_COUNT"

SEL_COUNT=$(python3 -c "import json; print(len(json.load(open('data/queries_selected.json'))))")
EXPECTED=$((POS_COUNT + NEG_COUNT))
if [ "$SEL_COUNT" -lt "$EXPECTED" ]; then
    echo "WARN: 选出的 baseline query 数 ($SEL_COUNT) < 期望 ($EXPECTED)，仍继续 Phase C 但表格不会满。" >&2
fi

# ============================================================
step 9 "Phase C: run 3 perturbations × ${SEL_COUNT} queries"
# ============================================================
python scripts/run_experiment.py \
    --config "$CONFIG" \
    --queries data/queries_selected.json \
    --bundle-types delete_top,add_irrelevant,replace_similar 2>&1 | tee /tmp/two_phase_C.log

# ============================================================
step 10 "Token / cost extrapolation + analyze"
# ============================================================
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY'
import json, os, pathlib
jsonl = pathlib.Path(os.environ['RESULTS_DIR']) / 'runs.jsonl'
runs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
print(f"\nRuns completed: {len(runs)}  (jsonl: {jsonl})")
by_type: dict[str, list] = {}
for r in runs:
    by_type.setdefault(r['bundle_type'], []).append(r)
for bt, rs in sorted(by_type.items()):
    rewards = [r.get('reward') for r in rs if r.get('reward') is not None]
    errs = sum(1 for r in rs if r.get('error_type'))
    skips = sum(1 for r in rs if r.get('skip_reason'))
    print(f"  {bt:18s} n={len(rs)}  reward_mean={sum(rewards)/len(rewards) if rewards else float('nan'):.2f}  err={errs}  skip={skips}")
toks = [r['token_count'] for r in runs if r.get('token_count')]
if toks:
    avg = sum(toks) / len(toks)
    cost_per_run = (avg * 0.95 * 3 + avg * 0.05 * 15) / 1_000_000
    print(f"\nAvg total tokens / run: {avg:.0f}")
    print(f"Estimated cost / run:   ${cost_per_run:.2f}")
    print(f"Total spend so far:     ${cost_per_run * len(runs):.2f}")
else:
    print("WARN: no token_count in any run; harbor/claude-code 上游问题，cost 估算跳过。")
PY

python scripts/analyze.py --config "$CONFIG" || {
    echo "WARN: analyze.py 失败，尝试无参数调用..." >&2
    python scripts/analyze.py || true
}

echo ""
echo "Two-phase done. Results at $JSONL"
echo "Phase A log: /tmp/two_phase_A.log"
echo "Phase C log: /tmp/two_phase_C.log"
