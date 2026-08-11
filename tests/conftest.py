import pathlib
import sys

# make src/ importable (mirrors the sys.path insert model_runs does at runtime)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
