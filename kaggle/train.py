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
MODE = "train"

# only used when MODE == 'sweep': (config_id, seed) pairs for THIS kernel.
SWEEP_RUN_LIST = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]

# find src path (handles both /gw-src-code/src and /gw-src-code layouts)
for candidate in ["/kaggle/input/gw-src-code/src", "/kaggle/input/gw-src-code"]:
    if Path(candidate).exists() and (Path(candidate) / "data").exists():
        src_path = Path(candidate)
        break
else:
    raise FileNotFoundError("Cannot find src code in /kaggle/input/gw-src-code")

print(f"Using src path: {src_path}")
print(f"Mode: {MODE}")
sys.path.insert(0, str(src_path))

# ============================================================================
#                              TRAINING
# ============================================================================

if MODE == "sweep":
    from model_runs import run_sweep
    run_sweep(SWEEP_RUN_LIST)
else:
    from model_runs import main
    main(mode=MODE)
