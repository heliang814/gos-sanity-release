"""从 harbor_jobs/ 原始 trial 重建干净的 runs.jsonl + 4×4 图。

用途：修复 _classify_error bug 后，用修复后的逻辑重新分类所有历史 trial，
排除 agent_setup_failed / agent_tokens=0 / reward=None 的污染数据，
按 (query_id, bundle_type) 聚合多次 trial 取均值。

输出：
  results_dir/runs_clean.jsonl     干净版（每个 trial 一行 + error_type）
  results_dir/aggregate_clean.json 按 cell 聚合后的 mean/std/n
  results_dir/bundle_vs_reward_clean.png  重画的 4×4 图
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent_runner import _classify_error, _read_harbor_result

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path('/mnt/c/Users/maxfo/Desktop/gos-sanity-results')
OUT_LOCAL = Path('/home/ghy/gos-sanity-results')
OUT_DESKTOP = RESULTS


def collect_trials() -> list[dict]:
    rows = []
    for job_dir in sorted(glob.glob(str(RESULTS / 'harbor_jobs/*/2026-*'))):
        parent = Path(job_dir).parent.name
        # parent 形如  <task>__<bundle_type>__<run_id>
        try:
            task, btype, run_id = parent.rsplit('__', 2)
        except ValueError:
            continue
        payload = _read_harbor_result(Path(job_dir))
        ar = payload.get('agent_result') or {}
        tok = (ar.get('n_input_tokens') or 0) + (ar.get('n_output_tokens') or 0)
        vr = payload.get('verifier_result') or {}
        reward = (vr.get('rewards') or {}).get('reward')
        err = _classify_error(payload, '')
        rows.append({
            'query_id': task,
            'bundle_type': btype,
            'run_id': run_id,
            'reward': reward,
            'token_count': tok,
            'error_type': err,
        })
    return rows


def aggregate(rows: list[dict]) -> dict:
    agg = defaultdict(list)
    for r in rows:
        if r['error_type'] is None and r['token_count'] > 0 and r['reward'] is not None:
            agg[(r['query_id'], r['bundle_type'])].append(r['reward'])
    out = {}
    for (q, bt), rs in agg.items():
        arr = np.array(rs, dtype=float)
        out.setdefault(q, {})[bt] = {
            'n': len(rs), 'mean': float(arr.mean()),
            'std': float(arr.std(ddof=0)), 'rewards': rs,
        }
    return out


def plot_matrix(agg: dict, out_path: Path) -> None:
    queries = ['pedestrian-traffic-counting', 'protein-expression-analysis',
               'crystallographic-wyckoff-position-analysis', 'xlsx-recover-data']
    btypes = ['gos_original', 'delete_top', 'add_irrelevant', 'replace_similar']
    short = {'pedestrian-traffic-counting': 'pedestrian',
             'protein-expression-analysis': 'protein',
             'crystallographic-wyckoff-position-analysis': 'crystal',
             'xlsx-recover-data': 'xlsx'}
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(queries))
    width = 0.2
    colors = ['#444', '#c33', '#3a7', '#37c']
    for i, bt in enumerate(btypes):
        means = [agg.get(q, {}).get(bt, {}).get('mean', np.nan) for q in queries]
        ns = [agg.get(q, {}).get(bt, {}).get('n', 0) for q in queries]
        bars = ax.bar(x + i * width - 1.5 * width, means, width, label=bt, color=colors[i])
        for bar, m, n in zip(bars, means, ns):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01,
                        f'{m:.2f}\nn={n}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([short[q] for q in queries])
    ax.set_ylabel('Mean reward (pass rate)')
    ax.set_title('GoS Sanity Check — 4×4 (cleaned, agent_setup_failed excluded)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f'[plot] {out_path}')


def main() -> None:
    rows = collect_trials()
    agg = aggregate(rows)

    # 写干净 jsonl
    for out in (OUT_LOCAL, OUT_DESKTOP):
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'runs_clean.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        (out / 'aggregate_clean.json').write_text(json.dumps(agg, ensure_ascii=False, indent=2))
        plot_matrix(agg, out / 'bundle_vs_reward_clean.png')

    # 控制台报告
    print('\n=== Cleaned 4×4 matrix ===')
    print(f"{'query':30s} {'bundle_type':18s} {'n':>3s} {'mean':>7s}  delta_vs_baseline")
    print('-' * 85)
    qs = ['pedestrian-traffic-counting', 'protein-expression-analysis',
          'crystallographic-wyckoff-position-analysis', 'xlsx-recover-data']
    for q in qs:
        cells = agg.get(q, {})
        base = cells.get('gos_original', {}).get('mean')
        for bt in ('gos_original', 'delete_top', 'add_irrelevant', 'replace_similar'):
            c = cells.get(bt)
            if c is None:
                print(f'{q[:28]:30s} {bt:18s}  --  MISSING')
                continue
            delta = (c['mean'] - base) if (base is not None and bt != 'gos_original') else 0
            d_str = f"  Δ={delta:+.3f}" if bt != 'gos_original' else ''
            print(f"{q[:28]:30s} {bt:18s} {c['n']:>3d} {c['mean']:>7.3f}{d_str}")
        print()


if __name__ == '__main__':
    main()
