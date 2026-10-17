"""
Run splitting and log relay for the multi-GPU sweep runner.
"""
from parallel_sweep import _Relay, split_round_robin


def test_round_robin_split_is_balanced_and_complete():
    runs = [(7, 0), (7, 1), (7, 2), (3, 0), (3, 1)]
    a, b = split_round_robin(runs, 2)
    assert a == [(7, 0), (7, 2), (3, 1)] and b == [(7, 1), (3, 0)]
    assert sorted(a + b) == sorted(runs)


def test_relay_prefixes_complete_lines_only(tmp_path, capsys):
    f = tmp_path / "out.txt"
    r = _Relay(f, "[gpu1] ")
    f.write_text("Epoch 1/20 - Val AUC: 0.84\nROW_JSON {\"a\"")
    r.pump()
    assert capsys.readouterr().out == "[gpu1] Epoch 1/20 - Val AUC: 0.84\n"
    # the half-written line is held until its newline arrives
    with open(f, "a") as h:
        h.write(": 1}\n")
    r.pump()
    assert capsys.readouterr().out == '[gpu1] ROW_JSON {"a": 1}\n'


def test_relay_drops_progress_bars(tmp_path, capsys):
    f = tmp_path / "out.txt"
    f.write_text("Epoch 1/2 [shard 2/2]:  50%|#####     | 5/10 [02:57<03:02, 36.56s/it, loss=0.77]\r"
                 "Logged 7_power_symmetric_match seed 0: AUC 0.8600\n")
    _Relay(f, "[gpu0] ").pump(final=True)
    assert capsys.readouterr().out == "[gpu0] Logged 7_power_symmetric_match seed 0: AUC 0.8600\n"
