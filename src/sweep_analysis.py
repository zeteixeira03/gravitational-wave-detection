"""
Tables and pre-registered contrasts, computed from the archived JSONL rows.

Every number in SWEEP_FINDINGS and ANALYSIS_PLAN section 8 must be recomputable
by this script from kaggle/sweep_log/*.jsonl; nothing is hand-transcribed.

Statistics (ANALYSIS_PLAN sections 0 and 8.3):
- per-config mean and sample sd (ddof=1) over seeds
- a contrast x - y is the mean over seeds of the paired difference x_s - y_s
- pooled sd = sqrt((s_x^2 + s_y^2) / 2) from the two configs' seed sds

Rows are grouped by stack (GPU model). Rows from different stacks are never
pooled or contrasted: config 1 seed 0 moved by -0.002 best_auc between P100 and
T4 (SWEEP_FINDINGS F10), the same size as the effects being measured. The 24
sweep rows predate the device field and all ran on P100.

Usage:
    python src/sweep_analysis.py [log_dir]
    python src/sweep_analysis.py --from-log <kernel_log.json> <out.jsonl>
        rebuild rows from a Kaggle kernel log (the API returns logs, while output
        files come from a host that may be unreachable). ROW_JSON lines are
        exact; older logs fall back to the printed 4-decimal summary and curves.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

METRICS = ('best_auc', 'auc')

# config 1 seed 0 as logged in v65 (git c703ac6), the reference for the
# instrument check in ANALYSIS_PLAN section 8.1
REFERENCE_ROW = {'config': 1, 'seed': 0, 'auc': 0.863110875492951, 'best_auc': 0.864665029661324}
MATCH_TOL = 1e-8

# section 8.3 contrasts: name -> (x, y), read as x - y
F3_CONTRASTS = {
    'S_4n (7 - 10)': (7, 10),
    'S_8n (9 - 3)': (9, 3),
    'W_ord (3 - 10)': (3, 10),
    'W_inv (9 - 7)': (9, 7),
}


def load_rows(log_dir) -> list[dict]:
    """All rows from every JSONL file in log_dir, tagged with their source file."""
    rows = []
    for path in sorted(Path(log_dir).glob('*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row['_file'] = path.name
                row['_config'] = int(str(row['config_id']).split('_')[0])
                row['_stack'] = row.get('device', 'Tesla P100')
                rows.append(row)
    return rows


_EPOCH = re.compile(r"Epoch (\d+)/\d+ - Loss: ([\d.]+) - Val Loss: ([\d.]+) - Val Acc: ([\d.]+) - Val AUC: ([\d.]+)")
_LOGGED = re.compile(r"Logged (\S+) seed (\d+): AUC ([\d.]+)\s+best ([\d.]+) @ epoch (\d+)/(\d+)\s+cond_params (\d+)\s+\(([\d.]+)h\)")
_MODEL = re.compile(r"Model: \w+ \(([\d,]+) parameters")
_DEVICE = re.compile(r'"device": "([^"]+)"')
_GIT = re.compile(r"Git hash: (\w+)|\"git_hash\": \"(\w+)\"")


def log_text(path) -> str:
    """Plain stdout text of a Kaggle kernel log (a JSON list of stream chunks)."""
    raw = Path(path).read_text()
    try:
        return ''.join(e.get('data', '') for e in json.loads(raw))
    except (json.JSONDecodeError, AttributeError):
        return raw


def rows_from_log(text: str, device: str | None = None) -> list[dict]:
    """
    Rows from a kernel log. ROW_JSON lines are the logged rows verbatim. For
    logs without them, each run is rebuilt from its epoch lines and its
    'Logged' summary line; every value is then rounded to 4 decimals, which
    the row records in 'precision'.
    """
    exact = [json.loads(line[len("ROW_JSON "):]) for line in text.splitlines() if line.startswith("ROW_JSON ")]
    if exact:
        return exact
    dev = device or (m.group(1) if (m := _DEVICE.search(text)) else None)
    git = next((g for m in _GIT.finditer(text) for g in m.groups() if g), None)
    rows, curve, params = [], [], None
    for line in text.splitlines():
        if (m := _MODEL.search(line)):
            params = int(m.group(1).replace(',', ''))
        elif (m := _EPOCH.search(line)):
            if int(m.group(1)) == 1:
                curve = []
            curve.append([float(x) for x in m.groups()[1:]])
        elif (m := _LOGGED.search(line)):
            label, seed, auc, best, b_ep, n_ep, cond, hours = m.groups()
            rows.append({
                'config_id': label, 'seed': int(seed), 'git_hash': git,
                'auc': float(auc), 'best_auc': float(best),
                'best_epoch': int(b_ep), 'epochs_run': int(n_ep),
                'total_params': params,
                'conditioning_params': int(cond), 'wall_clock_s': round(float(hours) * 3600, 1),
                'history': {
                    'train_loss': [c[0] for c in curve], 'val_loss': [c[1] for c in curve],
                    'val_acc': [c[2] for c in curve], 'val_auc': [c[3] for c in curve],
                },
                'device': dev, 'precision': 'log, 4 decimals',
            })
            curve = []
    return rows


def index_runs(rows) -> tuple[dict, list[str]]:
    """
    Map (config, seed) -> row. A pair logged more than once (a re-run) is kept
    once only if every copy agrees on both metrics to MATCH_TOL; disagreeing
    copies are reported, not averaged or silently dropped.
    """
    runs, conflicts = {}, []
    for row in rows:
        key = (row['_config'], row['seed'])
        if key not in runs:
            runs[key] = row
            continue
        first = runs[key]
        if any(abs(first[m] - row[m]) > MATCH_TOL for m in METRICS):
            conflicts.append(
                f"config {key[0]} seed {key[1]}: {first['_file']} "
                + ", ".join(f"{m} {first[m]:.10f}" for m in METRICS)
                + f" vs {row['_file']} "
                + ", ".join(f"{m} {row[m]:.10f}" for m in METRICS)
            )
    return runs, conflicts


def instrument_check(rows) -> dict | None:
    """
    Section 8.1: compare every config 1 seed 0 row logged after the reference
    against the reference values. The stack counts as unchanged only if every
    re-run matches. None if no re-run exists yet.
    """
    reruns = [r for r in rows if r['_config'] == REFERENCE_ROW['config']
              and r['seed'] == REFERENCE_ROW['seed']
              and not r.get('git_hash', '').startswith('c703ac6')]
    if not reruns:
        return None
    checks = [{
        'file': r['_file'],
        'git_hash': r.get('git_hash', ''),
        'diffs': {m: r[m] - REFERENCE_ROW[m] for m in METRICS},
    } for r in reruns]
    return {
        'reruns': checks,
        'unchanged': all(abs(d) <= MATCH_TOL for c in checks for d in c['diffs'].values()),
    }


def config_values(runs, config, metric) -> dict[int, float]:
    """seed -> metric for one config."""
    return {s: r[metric] for (c, s), r in runs.items() if c == config}


def summary_table(runs) -> list[dict]:
    out = []
    for config in sorted({c for c, _ in runs}):
        entry = {'config': config, 'name': next(r['config_id'] for (c, _), r in runs.items() if c == config)}
        for m in METRICS:
            v = np.array(list(config_values(runs, config, m).values()))
            entry[m] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float('nan'), len(v))
        entry['total_params'] = next(r['total_params'] for (c, _), r in runs.items() if c == config)
        out.append(entry)
    return out


def contrast(runs, x, y, metric) -> dict | None:
    """Paired x - y over the seeds both configs share, with the pooled sd."""
    vx, vy = config_values(runs, x, metric), config_values(runs, y, metric)
    seeds = sorted(set(vx) & set(vy))
    if len(seeds) < 2:
        return None
    d = [vx[s] - vy[s] for s in seeds]
    sx = np.std([vx[s] for s in seeds], ddof=1)
    sy = np.std([vy[s] for s in seeds], ddof=1)
    pooled = float(np.sqrt((sx ** 2 + sy ** 2) / 2))
    mean = float(np.mean(d))
    return {'seeds': seeds, 'diffs': d, 'mean': mean, 'pooled_sd': pooled, 'ratio': mean / pooled}


def f3_verdict(c: dict) -> str:
    """Apply the section 8.3 decision rules to computed contrasts (best_auc and auc)."""
    def get(name, m):
        return c.get((name, m))

    def above(name):
        return all(get(name, m) and get(name, m)['mean'] > get(name, m)['pooled_sd'] for m in METRICS)

    def within(name):
        return all(get(name, m) and abs(get(name, m)['mean']) <= get(name, m)['pooled_sd'] for m in METRICS)

    def below(name):
        return all(get(name, m) and get(name, m)['mean'] < -get(name, m)['pooled_sd'] for m in METRICS)

    if any(get(n, m) is None for n in ('S_4n (7 - 10)', 'W_ord (3 - 10)') for m in METRICS):
        return "not evaluable: config 10 rows missing"
    if within('S_4n (7 - 10)') and below('W_ord (3 - 10)'):
        return "config 7's gain over 3 is capacity, not symmetry (rule fires)"
    if any(get('S_8n (9 - 3)', m) is None for m in METRICS):
        return "capacity rule does not fire; symmetry rule waits for config 9 (week 2)"
    if above('S_4n (7 - 10)') and above('S_8n (9 - 3)'):
        return "symmetry helps (rule fires)"
    return "no rule fires: report as observed, no causal reading"


def main(log_dir):
    rows = load_rows(log_dir)
    print(f"{len(rows)} rows from {log_dir}\n")
    check = instrument_check(rows)
    if check is None:
        print("Instrument check: no config 1 seed 0 re-run logged yet\n")
    else:
        verdict = "UNCHANGED" if check['unchanged'] else "CHANGED"
        print(f"Instrument check: stack {verdict}")
        for c in check['reruns']:
            print(f"  {c['file']} ({c['git_hash'][:7]}): "
                  + ", ".join(f"d{m} {d:+.3e}" for m, d in c['diffs'].items()))
        print()
    for stack in sorted({r['_stack'] for r in rows}):
        report_stack(stack, [r for r in rows if r['_stack'] == stack])


def report_stack(stack, rows):
    """Table, contrasts and verdict from the rows of one stack only."""
    runs, conflicts = index_runs(rows)
    print("=" * 96)
    print(f"STACK {stack}: {len(rows)} rows, {len(runs)} distinct (config, seed) pairs")
    print("=" * 96)
    for line in conflicts:
        print(f"CONFLICT {line}")
    if conflicts:
        print()

    print(f"{'config':<28}{'best_auc mean':>14}{'sd':>9}{'auc mean':>11}{'sd':>9}{'n':>3}{'params':>11}")
    for e in summary_table(runs):
        b, a = e['best_auc'], e['auc']
        print(f"{e['name']:<28}{b[0]:>14.5f}{b[1]:>9.5f}{a[0]:>11.5f}{a[1]:>9.5f}{b[2]:>3}{e['total_params']:>11,}")

    print("\nF3 contrasts (ANALYSIS_PLAN 8.3), paired by seed")
    computed = {}
    for name, (x, y) in F3_CONTRASTS.items():
        for m in METRICS:
            c = contrast(runs, x, y, m)
            computed[(name, m)] = c
            if c is None:
                print(f"  {name:<16}{m:<9} not available")
            else:
                diffs = ", ".join(f"{d:+.5f}" for d in c['diffs'])
                print(f"  {name:<16}{m:<9} mean {c['mean']:+.5f}  pooled sd {c['pooled_sd']:.5f}  "
                      f"({c['ratio']:+.2f} sd)  per seed [{diffs}]")
    print(f"\nF3 verdict: {f3_verdict(computed)}\n")


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--from-log':
        rebuilt = rows_from_log(log_text(sys.argv[2]))
        with open(sys.argv[3], 'w') as f:
            for r in rebuilt:
                f.write(json.dumps(r) + "\n")
        print(f"{len(rebuilt)} rows written to {sys.argv[3]}")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'kaggle' / 'sweep_log')
