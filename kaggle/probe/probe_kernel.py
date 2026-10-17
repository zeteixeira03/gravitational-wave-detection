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

# new kernels mount inputs at a deeper path than gw-training's
# /kaggle/input/gw-src-code, so find the source by its files
hits = sorted(Path("/kaggle/input").rglob("model_runs.py"))
if not hits:
    for p in sorted(Path("/kaggle/input").rglob("*"))[:50]:
        print(p)
    raise FileNotFoundError("Cannot find model_runs.py under /kaggle/input")
src_path = hits[0].parent

print(f"Using src path: {src_path}")
sys.path.insert(0, str(src_path))

from probe import run_probe
run_probe(config_id=1, seed=0)
