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
"""
import sys
from pathlib import Path

# ============================================================================
#                              SETUP
# ============================================================================

# mode: 'train' for full run, 'lr_test' for LR range test.
MODE = "train"

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

from model_runs import main
main(mode=MODE)
