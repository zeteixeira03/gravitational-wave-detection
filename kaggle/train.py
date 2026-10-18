"""
Kaggle training script - uses preprocessed .pt tensors for fast training.

The tensor dataset contains signals that have been preprocessed locally:
- Bandpass filtered (20-500 Hz)
- Whitened using noise PSD
- Tukey windowed
- Normalized

Hyperparameters are defined in src/model_runs.py main().

Set MODE to 'lr_test' to run a 1-epoch LR range test instead of full training,
then flip back to 'train' before pushing the actual run.

Set MODE to 'sweep' to run a slice of the pre-registered sky-readout sweep.
SWEEP_RUN_LIST is the list of (config_id, seed) pairs this kernel runs, in
order. Split the pairs across the two concurrent Kaggle sessions so no pair
runs twice. Config ids and seeds are defined in model_runs.SWEEP_CONFIGS and
ANALYSIS_PLAN. Results append to <output>/sweep_results.jsonl.
"""
import sys
from pathlib import Path

# ============================================================================
#                              SETUP
# ============================================================================

# mode: 'train' full run | 'lr_test' LR range test | 'sweep' sky-readout sweep.
MODE = "sweep"

# only used when MODE == 'sweep': (config_id, seed) pairs for THIS kernel.
# With 2 GPUs the list is split round-robin, one process per GPU; a T4 run takes
# ~0.75h. F10 moved the project to T4, so configs 3 and 7 are re-run there
# before any F3 contrast (ANALYSIS_PLAN 8.1). v71 ran config 7 x3 and (3,0).
# v72 ran config 3 seeds 1-2. Next quota week (from 2026-10-24): ANALYSIS_PLAN
# section 9 in full plus the first config 9 seed; see SWEEP_FINDINGS section 6.
SWEEP_RUN_LIST = [(11, 0), (11, 1), (11, 2), (1, 1), (1, 2), (9, 0)]

# find src path by its files: kernels created after 2026-08 mount inputs below
# /kaggle/input/datasets/..., older ones at /kaggle/input/<slug>
hits = sorted(Path("/kaggle/input").rglob("model_runs.py"))
if not hits:
    raise FileNotFoundError("Cannot find model_runs.py under /kaggle/input")
src_path = hits[0].parent

print(f"Using src path: {src_path}")
print(f"Mode: {MODE}")
sys.path.insert(0, str(src_path))

# ============================================================================
#                              TRAINING
# ============================================================================

if MODE == "sweep":
    import torch
    # one training process per GPU: the quota counts session hours, so two T4s
    # halve the cost of each run (see src/parallel_sweep.py)
    if torch.cuda.device_count() >= 2 and len(SWEEP_RUN_LIST) > 1:
        from parallel_sweep import run_parallel
        run_parallel(SWEEP_RUN_LIST, "/kaggle/working", n_gpu=2)
    else:
        from model_runs import run_sweep
        run_sweep(SWEEP_RUN_LIST)
else:
    from model_runs import main
    main(mode=MODE)
