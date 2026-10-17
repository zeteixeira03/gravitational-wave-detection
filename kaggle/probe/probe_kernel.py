"""
Kaggle entry point for the F2 redundancy probe and the F9 shard check.

Trains config 1 seed 0 at the sweep recipe, logs its row to
<output>/sweep_results.jsonl like any sweep run, then probes its pooled
features. Results go to <output>/probe_results.json. See src/probe.py and
ANALYSIS_PLAN section 8.

The docker image is pinned to the one gw-training v69 ran on, so this run uses
the same stack as the 24 sweep rows. Config 1 seed 0 is also in that log: if
the stack is unchanged, the harness is deterministic (SWEEP_FINDINGS F7) and
this row must reproduce the logged one.
"""
import sys
from pathlib import Path

for candidate in ["/kaggle/input/gw-src-code/src", "/kaggle/input/gw-src-code"]:
    if Path(candidate).exists() and (Path(candidate) / "data").exists():
        src_path = Path(candidate)
        break
else:
    raise FileNotFoundError("Cannot find src code in /kaggle/input/gw-src-code")

print(f"Using src path: {src_path}")
sys.path.insert(0, str(src_path))

from probe import run_probe
run_probe(config_id=1, seed=0)
