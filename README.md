# gos-sanity

GoS bundle 敏感性 sanity check 实验仓。给定一组 query，测量 GoS PPR 选出的 skill bundle 与对其做 3 类扰动后的 bundle 在 Harbor + claude-code agent 上的 reward 差异，验证"bundle 选择对 agent 表现是否真有影响"。

设计为 **4 query × 4 bundle_type = 16 cell**，每 cell 跑 n 次 agent rollout 取均值。

## 目录结构

```
gos-sanity-release/
├── README.md                  本文件
├── HANDOFF.md                 完整运行手册（环境、安装、API key、踩坑）
├── pyproject.toml             包定义
├── setup_and_smoke.sh         一键环境检查 + smoke 试验
├── configs/
│   └── experiment.yaml        agent/perturbation/路径配置
├── data/
│   ├── queries.json           30 个候选 query 池（Phase A 输入）
│   ├── queries_baseline.json  Phase A 跑过 baseline 的 query
│   ├── queries_phaseC.json    Phase C 选出的 query
│   ├── queries_selected.json  本次 4×4 实验最终选定的 4 query
│   ├── queries_pedestrian.json 单 query 文件（topup 用）
│   ├── queries_protein.json   单 query 文件（topup 用）
│   └── risky_tasks.json       已知不稳定/被踢出的 task
├── src/
│   ├── agent_runner.py        单次 trial: stage bundle → harbor run → 解析 result.json
│   ├── gos_interface.py       调 GoS PPR retrieval + query embedding
│   └── perturbations.py       delete_top / add_irrelevant / replace_similar
├── scripts/
│   ├── run_experiment.py      主驱动: queries × bundle_types → trials
│   ├── run_two_phase.sh       Phase A baseline + Phase C 扰动
│   ├── run_topup.sh           当前补 n 用的串行脚本
│   ├── reaggregate.py         从 results/harbor_jobs/* 重新聚合 + 重画图
│   ├── analyze.py             读 runs.jsonl 出汇总
│   ├── screen_smoke_queries.py / select_baseline_queries.py / select_queries.py
│   │                          Phase B 候选筛选三件套
└── results/
    ├── runs.jsonl             每 trial 一行（query/bundle/skill_ids/reward/error_type/...）
    ├── runs_clean.jsonl       reaggregate 重生成、过滤过 error 的版本
    ├── aggregate_clean.json   按 (query, bundle_type) 聚合的 mean/std/n/rewards
    ├── default_bundles.json   各 query 的原始 GoS bundle + 全库 PPR
    ├── bundle_vs_reward_clean.png   4×4 reward bar plot
    ├── bundles_4x4.md         16 cell bundle 详表 + 横向观察 + 结论（主报告）
    └── topup.log              topup 阶段 stdout 日志（时间戳 + harbor 输出）
```

## 当前实验状态

### 4×4 reward 矩阵

| query | gos_original | delete_top | add_irrelevant | replace_similar |
|---|---|---|---|---|
| pedestrian | 0.084±0.076 (n=4) | 0.090±0.140 (n=4) Δ=+0.006 | 0.103±0.059 (n=2) Δ=+0.019 | 0.044±0.049 (n=3) Δ=−0.040 |
| protein | 0.500±0.500 (n=4) | 0.250±0.433 (n=4) Δ=−0.250 | 0.000 (n=4) Δ=−0.500 | 0.000 (n=4) Δ=−0.500 |
| crystal | 0.550 (n=1) | 0.550 (n=1) | 0.550 (n=1) | 0.550 (n=1) |
| xlsx | 0.000 (n=2) | 0.000 (n=1) | 0.000 (n=1) | 0.000 (n=1) |

### 关键发现（详见 `results/bundles_4x4.md`）

1. **protein 是当前唯一可声称扰动有效的 cell**：单调 0.500→0.250→0.000→0.000，方向符合 GoS 假设；但 std=0.5、n=4 下 95% CI 仍宽（gos vs add p≈0.06），统计上 underpowered。
2. **pedestrian 在 n=4 下扰动效应消失**：所有 \|Δ\|≤0.04，单 cell std 0.05–0.14，远大于 cell 间差异 → "GoS bundle 优于扰动"在 n=4 下不成立。
3. **crystal/xlsx 无判别力**：crystal 4 cell 全 0.55（常数任务）、xlsx 4 cell 全 0（任务对当前 agent 太难）。
4. **pedestrian add_irrelevant/replace_similar 有 3 个 trial 撞 1h `cfg.agent.timeout_s` 上限**，导致这两 cell 实际只跑到 n=2 / n=3。

### 已知 bug 修复

- `src/agent_runner._classify_error` 之前先看 reward 后看 exception_info，导致 `agent_setup_failed`（agent 进程崩溃 + verifier 看到空 solution 给 0.0）被误归类为正常完成、污染 jsonl。已改为优先检查 `exception_info.exception_type`，再按 token=0 区分 setup_failed / agent_failed。
- `scripts/reaggregate.py` 的 `RESULTS` 路径已修正为 Desktop（之前写死本地缓存路径会漏读新数据）。

## 快速复现

完整环境配置见 `HANDOFF.md`。简版：

```bash
# 1. 安装
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. 环境变量（.env.local 里放 ANTHROPIC_API_KEY=sk-ant-...）
echo 'ANTHROPIC_API_KEY=sk-ant-xxx' > .env.local

# 3. Docker 必须可访问（Linux 检查）
docker ps  # 不能报 permission denied

# 4. 跑全部 4×4
python -m scripts.run_experiment --queries data/queries_selected.json

# 5. 补 n（pedestrian/protein 单 query 单 bundle_type）
bash scripts/run_topup.sh

# 6. 重聚合 + 重画图
python scripts/reaggregate.py
```

预算：单 trial 5–60min（视任务复杂度），claude-sonnet-4-5 单 trial ~$0.3，全 16 cell × n=4 ≈ $20、6–10h 串行。

## 后续建议

- 想让 protein 信号统计显著（gos vs add p<0.05）：每 cell 补到 n≥8。
- 想看 pedestrian 真实信号：补 add_irrelevant 至 n=4（差 2）+ replace_similar 至 n=4（差 1），但当前数据已暗示扰动效应在噪声内，扩 n 到 8 也大概率不显著。
- crystal/xlsx 应换 query —— 这两个对当前 agent 是常数任务，不携带 bundle 选择信号。
- 长 timeout 任务（pedestrian）建议把 `cfg.agent.timeout_s` 从 3600 推到 5400 或 7200，避免多 trial 截断。
