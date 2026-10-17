"""
Decision-rule logic of sweep_analysis on synthetic rows with known answers.
"""
import json

from sweep_analysis import REFERENCE_ROW, contrast, f3_verdict, index_runs, instrument_check, load_rows, F3_CONTRASTS, METRICS


def _write(tmp_path, rows, name="v.jsonl"):
    with open(tmp_path / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rows(config, values, git="abc"):
    # values: per seed (best_auc, auc)
    return [{'config_id': f"{config}_x", 'seed': s, 'best_auc': b, 'auc': a,
             'total_params': 1, 'git_hash': git} for s, (b, a) in enumerate(values)]


def _verdict(tmp_path, cfg_values):
    rows = [r for c, v in cfg_values.items() for r in _rows(c, v)]
    _write(tmp_path, rows)
    runs, _ = index_runs(load_rows(tmp_path))
    computed = {(n, m): contrast(runs, x, y, m) for n, (x, y) in F3_CONTRASTS.items() for m in METRICS}
    return f3_verdict(computed)


def test_paired_contrast_and_pooled_sd(tmp_path):
    _write(tmp_path, _rows(7, [(0.864, 0.0), (0.863, 0.0), (0.865, 0.0)])
           + _rows(10, [(0.863, 0.0), (0.862, 0.0), (0.864, 0.0)]))
    runs, _ = index_runs(load_rows(tmp_path))
    c = contrast(runs, 7, 10, 'best_auc')
    assert abs(c['mean'] - 0.001) < 1e-12
    assert abs(c['pooled_sd'] - 0.001) < 1e-12   # both sds are 0.001


def test_capacity_rule_fires(tmp_path):
    # 7 == 10 within noise, and 10 beats 3 clearly
    v = _verdict(tmp_path, {
        3: [(0.860, 0.858), (0.861, 0.859), (0.862, 0.860)],
        7: [(0.864, 0.863), (0.865, 0.864), (0.866, 0.865)],
        10: [(0.8645, 0.8632), (0.8645, 0.8642), (0.8655, 0.8648)],
    })
    assert v.startswith("config 7's gain over 3 is capacity")


def test_symmetry_waits_for_config_9(tmp_path):
    # 7 clearly above 10: capacity rule cannot fire, config 9 absent
    v = _verdict(tmp_path, {
        3: [(0.860, 0.858), (0.861, 0.859), (0.862, 0.860)],
        7: [(0.866, 0.865), (0.867, 0.866), (0.868, 0.867)],
        10: [(0.860, 0.859), (0.861, 0.860), (0.862, 0.861)],
    })
    assert "config 9" in v


def test_conflicting_duplicates_are_reported(tmp_path):
    _write(tmp_path, _rows(1, [(0.8, 0.8)]), "a.jsonl")
    _write(tmp_path, _rows(1, [(0.8, 0.7)]), "b.jsonl")
    _, conflicts = index_runs(load_rows(tmp_path))
    assert len(conflicts) == 1


def test_instrument_check(tmp_path):
    ref = (REFERENCE_ROW['best_auc'], REFERENCE_ROW['auc'])
    _write(tmp_path, _rows(1, [ref], git="c703ac6x") + _rows(1, [ref], git="4d90382"))
    assert instrument_check(load_rows(tmp_path))['unchanged'] is True
    _write(tmp_path, _rows(1, [(ref[0] + 1e-4, ref[1])], git="4d90382"), "c.jsonl")
    assert instrument_check(load_rows(tmp_path))['unchanged'] is False
