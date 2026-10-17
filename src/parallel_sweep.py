"""
Run a sweep slice on several GPUs at once, one training process per GPU.

Why: the Kaggle quota counts session hours, not GPU hours, and a session now
gets two T4s. One process per GPU halves the quota cost of each run.

How:
- The run list is split round-robin, so each GPU gets a balanced share and a
  kernel cut short still leaves complete seeds for the first configs.
- Each child sees one GPU (CUDA_VISIBLE_DEVICES) and calls run_sweep on its
  share, with its own JSONL log and its own stdout file.
- The parent is the only writer to the kernel log: it relays each child's
  complete lines with a "[gpuK] " prefix. The kernel log is the one output the
  Kaggle API always returns, so its lines must not interleave mid-line.
- The second child starts only after the first has loaded its validation data,
  so the two load-time memory peaks never overlap.

A child's runs are as reproducible as in a single-process kernel: every run
seeds itself (set_seed) and uses only its own GPU.
"""
import subprocess
import sys
import time
from pathlib import Path

# printed by train_from_tensors once the validation shards are in memory
LOADED_MARKER = "Training shards:"
# tqdm progress bars ("50%|#####     | 5/10 [...]"): kept in the child's file,
# not relayed to the kernel log
_NOISE = ("%|",)


def split_round_robin(run_list, n):
    """[(c, s), ...] -> n lists, item i to list i % n."""
    return [run_list[k::n] for k in range(n)]


def _child_code(src_dir, runs, log_path, base_overrides):
    lines = [
        "import sys",
        f"sys.path.insert(0, {str(src_dir)!r})",
        "import model_runs",
    ]
    if base_overrides:
        # smoke tests only: shrink the recipe so a CPU test finishes quickly
        lines.append(f"model_runs.SWEEP_BASE.update({base_overrides!r})")
    lines.append(f"model_runs.run_sweep({runs!r}, log_path={str(log_path)!r})")
    return "\n".join(lines)


class _Relay:
    """Prints a child's new complete lines to our stdout with a prefix."""

    def __init__(self, path, prefix):
        self.path, self.prefix, self.pos, self.partial = Path(path), prefix, 0, ""

    def pump(self, final=False):
        if not self.path.exists():
            return ""
        with open(self.path, "r", errors="replace") as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        text = self.partial + chunk.replace("\r", "\n")
        lines = text.split("\n")
        self.partial = "" if final else lines.pop()
        for line in lines:
            if line.strip() and not any(n in line for n in _NOISE):
                print(f"{self.prefix}{line}", flush=True)
        return text


def run_parallel(run_list, output_dir, n_gpu=2, base_overrides=None, poll_s=20):
    """
    Run ``run_list`` across ``n_gpu`` child processes. Returns the per-GPU
    JSONL paths. Raises if any child exits non-zero, after relaying its output.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parent
    shares = [s for s in split_round_robin(run_list, n_gpu) if s]
    print(f"Parallel sweep on {len(shares)} GPUs: " +
          "; ".join(f"gpu{k} {s}" for k, s in enumerate(shares)), flush=True)

    procs, relays, logs = [], [], []
    for k, share in enumerate(shares):
        log_path = output_dir / f"sweep_results_gpu{k}.jsonl"
        out_path = output_dir / f"stdout_gpu{k}.txt"
        logs.append(log_path)
        relays.append(_Relay(out_path, f"[gpu{k}] "))
        env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=str(k), PYTHONUNBUFFERED="1")
        with open(out_path, "w") as out:
            procs.append(subprocess.Popen(
                [sys.executable, "-u", "-c", _child_code(src_dir, share, log_path, base_overrides)],
                stdout=out, stderr=subprocess.STDOUT, env=env,
            ))
        # stagger: the next child starts once this one holds its val data
        if k < len(shares) - 1:
            while procs[-1].poll() is None and LOADED_MARKER not in out_path.read_text(errors="replace"):
                time.sleep(poll_s)
                for r in relays:
                    r.pump()

    while any(p.poll() is None for p in procs):
        time.sleep(poll_s)
        for r in relays:
            r.pump()
    for r in relays:
        r.pump(final=True)

    codes = [p.returncode for p in procs]
    print(f"Parallel sweep finished, exit codes {codes}", flush=True)
    if any(codes):
        raise RuntimeError(f"child process failed: exit codes {codes}")
    return logs
