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


def test_verdict_names_missing_contrast(tmp_path):
    # config 3 at one seed only: W_ord cannot be formed, S_4n can
    v = _verdict(tmp_path, {
        3: [(0.860, 0.858)],
        7: [(0.864, 0.863), (0.865, 0.864), (0.866, 0.865)],
        10: [(0.8645, 0.8632), (0.8645, 0.8642), (0.8655, 0.8648)],
    })
    assert v.startswith("not evaluable") and "W_ord" in v and "S_4n" not in v


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


def test_rows_from_log_fallback_and_exact():
    from sweep_analysis import rows_from_log
    # lines copied from the gw-probe v2 log (no ROW_JSON yet)
    text = "\n".join([
        ' "device": "Tesla T4",',
        'Git hash: b88df30693d88aeeeb0cef6678d96f24706420ed',
        'Model: DIYModel (4,556,010 parameters, conditioning path 0)',
        'Epoch 1/20 - Loss: 0.5299 - Val Loss: 0.5439 - Val Acc: 0.6851 - Val AUC: 0.8409 - LR: 6.67e-04 *',
        'Epoch 2/20 - Loss: 0.4676 - Val Loss: 0.5066 - Val Acc: 0.7520 - Val AUC: 0.8563 - LR: 1.33e-03 *',
        'Logged 1_none_concat seed 0: AUC 0.8617  best 0.8626 @ epoch 8/11  cond_params 0  (0.73h)',
    ])
    (row,) = rows_from_log(text)
    assert (row['config_id'], row['seed'], row['auc'], row['best_auc']) == ('1_none_concat', 0, 0.8617, 0.8626)
    assert row['device'] == 'Tesla T4' and row['git_hash'].startswith('b88df30')
    assert row['history']['val_auc'] == [0.8409, 0.8563]
    assert row['value_precision'] == 'log, 4 decimals' and row['total_params'] == 4556010
    # a ROW_JSON line wins and is taken verbatim
    exact = {'config_id': '10_x', 'seed': 2, 'auc': 0.86123456789, 'best_auc': 0.8621}
    assert rows_from_log(text + "\nROW_JSON " + json.dumps(exact)) == [exact]
    # logs from before the stack report name the GPU only in check_gpu's output
    old_log = text.replace(' "device": "Tesla T4",', 'GPU(s) detected: 2\n  - Tesla T4\n  - Tesla T4')
    assert rows_from_log(old_log)[0]['device'] == 'Tesla T4'


def test_stacks_are_not_pooled(tmp_path):
    rows = _rows(1, [(0.86, 0.86)]) + [dict(r, device='Tesla T4', best_auc=0.85) for r in _rows(1, [(0.85, 0.85)])]
    _write(tmp_path, rows)
    stacks = {r['_stack'] for r in load_rows(tmp_path)}
    assert stacks == {'Tesla P100', 'Tesla T4'}
