# gos-sanity 运行手册

GoS bundle 敏感性 sanity check：20 query × 4 bundle = 80 次 agent rollout，验证 skill bundle 变化是否影响 reward。完整跑通约 1–2 小时，成本 ~$200–300。

---

## 0. 机器要求

**必须满足**：
- macOS（推荐 Sonoma+）或 Ubuntu 22.04+
- **Docker daemon 在跑**（Mac: OrbStack 或 Docker Desktop；Linux: 原生 docker）
- 8GB+ RAM，10GB+ 可用磁盘
- **能直连 `api.anthropic.com`**（国内 IP 会被 geo-block 返 403，必须有 VPN 出境到美国/日本/欧洲）
- Python 3.10–3.12（**不要用 3.13**，fast-graphrag 装不上）

**Mac 额外坑**：
- 项目**不要放在 `~/Desktop` 或 `~/Documents`** —— 这两个目录可能被 iCloud 同步，会导致 file I/O 错乱。放 `~/code/` 之类的非同步目录。
- Xcode Command Line Tools 必须完整（`ls /Library/Developer/CommandLineTools/usr/include/c++/v1/iostream` 必须有输出，否则 hnswlib 编不过）。坏了重装：`sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`

---

## 1. 一键安装运行

```bash
# Step 1: clone（非 iCloud 同步目录）
mkdir -p ~/code && cd ~/code
git clone git@github.com:ANTI-Tony/sk.git gos-sanity
# 没 SSH key 的话：git clone https://github.com/ANTI-Tony/sk.git gos-sanity

# Step 2: 验证 Docker
docker info | head -3
# 必须看到 "Server:" + "Server Version: ..."；看到 "Cannot connect" 先打开 OrbStack/Docker Desktop

# Step 3: 验证 Anthropic API 可达（关键）
curl -sI --max-time 10 https://api.anthropic.com/v1/messages | head -1
# HTTP/2 401 = 通（dummy 请求被拒，但网络通）
# HTTP/2 403 = geo-block，必须开 VPN 后再做

# Step 4: 配置 API keys
cd gos-sanity
cat > .env.local <<'EOF'
ANTHROPIC_API_KEY=<向 Tony 索取>
OPENAI_API_KEY=<向 Tony 索取>
EOF

# Step 5: 一键跑（脚本会自动装 uv, harbor, GoS deps，下数据，跑 4 runs smoke test）
bash setup_and_smoke.sh
```

`setup_and_smoke.sh` 自动做的事：
- Clone graph-of-skills 到同级目录
- 装 uv + harbor
- 下数据（skills_200 + workspace + 87 SkillsBench tasks，~600MB）
- 选 1 个 query 跑 4 个 bundle 做 smoke test
- 算 token 和 80 runs 外推成本

---

## 2. Smoke test 后的判断

跑完输出长这样：

```
Runs completed: 4
  gos_original         reward=1.0 tokens=85000  time=540s err=None
  delete_top           reward=0.0 tokens=72000  time=480s err=None
  add_irrelevant       reward=1.0 tokens=91000  time=620s err=None
  replace_similar      reward=0.0 tokens=88000  time=590s err=None

Avg total tokens / run: 84000
Estimated cost / run:    $0.27
Extrapolate 80 runs:     $22

OK: <= $240. Safe to top up balance and run full.
```

**4 个判断**：
1. **reward 不是 null** → ✅ 整套 pipeline 工作
2. **reward 在条件间有差异**（不是 4 个全 0 或全 1）→ ✅ 有 sensitivity 信号
3. **Extrapolate 80 runs cost** 在合理范围（< $400）→ ✅ 全量预算 OK
4. **error_type 都是 None**（或只 1 个）→ ✅ 没系统问题

**全部 ✅ 才跑全量**：

```bash
# 把 data/queries.json 还原成 20 个 query
python scripts/select_queries.py
# 跑全 80 runs（1–2 小时）
python scripts/run_experiment.py
# 出报告 + 柱状图
python scripts/analyze.py
```

最终结果在 `/tmp/gos-sanity-results/runs.jsonl` 和 `/tmp/gos-sanity-results/bundle_vs_reward.png`。

---

## 3. 常见问题排查

### Smoke test 4 个 reward 全 null + error_type=agent_failed
Harbor 在 3 分钟时 cancel 了 agent。打开 `configs/experiment.yaml`，把 `harbor_timeout_multiplier: 5` 改成 `10` 或 `20`，重跑。

### `pip install hnswlib` 报 "iostream not found"（Mac）
Xcode CLT 坏了：
```bash
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install   # 弹窗点 Install，等装完
```

### 数据下载失败
脚本默认走 hf-mirror 和 gh-proxy 备份源，国内能用。如果 HF 直连慢，第一次跑会自动 fallback。

### `_append_jsonl` 报 errno 60（Mac）
项目在 iCloud 同步目录里，把它挪到 `~/code/`。

### 跑到一半要停
直接 Ctrl+C。每个 bundle 跑完会立刻 append 到 `runs.jsonl`，已完成的不会丢，重跑 `python scripts/run_experiment.py` 会从 cache 接着来。

---

## 4. 联系

跑完结果 + log 发给 Tony：
- `/tmp/gos-sanity-results/runs.jsonl`（4 行 JSON 一行一个 run）
- `/tmp/gos-sanity-results/bundle_vs_reward.png`（柱状图）
- 控制台 step 9 的输出截图
