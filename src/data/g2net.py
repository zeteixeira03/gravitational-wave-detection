from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SAMPLE_DATASET_SLUG = "zeteixeira/g2net-exploration-sample"


# ---------------------------------------------------------------------
#                        environment detection
# ---------------------------------------------------------------------

def is_kaggle() -> bool:
    """Check if running in Kaggle environment."""
    return Path("/kaggle").exists()


def get_output_dir() -> Path:
    """
    Get the appropriate output directory for the current environment.

    Returns /kaggle/working on Kaggle, or project_root locally.
    """
    if is_kaggle():
        return Path("/kaggle/working")
    return find_project_root()


# ---------------------------------------------------------------------
#                   project / dataset discovery
# ---------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find the project root as the first directory (walking upwards) that
    contains a 'src' folder.

    Parameters
    ----------
    start : Path, optional
        Starting directory. Defaults to current working directory.

    Returns
    -------
    Path
        Path to the detected project root.
    """
    if start is None:
        start = Path.cwd()

    for p in [start, *start.parents]:
        if (p / "src").exists():
            return p

    # fallback: current working directory
    return Path.cwd()


def sample_dataset_dir(project_root: Optional[Path] = None) -> Path:
    """Return the local path where the exploration sample is stored."""
    if project_root is None:
        project_root = find_project_root()
    return project_root / "data" / "g2net-exploration-sample"


def _is_valid_dataset(path: Path) -> bool:
    """Check whether *path* looks like a usable G2Net dataset (full or sample)."""
    return path.exists() and (path / "train").is_dir()


def download_sample_dataset(project_root: Optional[Path] = None) -> Path:
    """
    Download the exploration sample (~100 MB) via the Kaggle CLI.

    Parameters
    ----------
    project_root : Path, optional
        Project root. Resolved automatically if omitted.

    Returns
    -------
    Path
        Directory containing the downloaded sample dataset.
    """
    dest = sample_dataset_dir(project_root)
    if _is_valid_dataset(dest):
        return dest

    print(f"Downloading exploration sample from Kaggle ({SAMPLE_DATASET_SLUG})...")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", SAMPLE_DATASET_SLUG,
         "--unzip", "-p", str(dest)],
        check=True,
    )

    if not _is_valid_dataset(dest):
        raise RuntimeError(
            f"Download succeeded but {dest} does not contain a valid dataset. "
            "Check the Kaggle dataset structure."
        )

    print(f"Sample dataset ready at {dest}")
    return dest


def find_dataset_dir(project_root: Optional[Path] = None) -> Path:
    """
    Find the G2Net dataset directory.

    Search order:
    1. Kaggle input directory (when running on Kaggle)
    2. Environment variable G2NET_DATASET_PATH
    3. External drive (local development)
    4. Full dataset under project_root/data/
    5. Exploration sample under project_root/data/g2net-exploration-sample

    Parameters
    ----------
    project_root : Path, optional
        Project root directory. If None, it is resolved via find_project_root().

    Returns
    -------
    Path
        Path to the dataset directory

    Raises
    ------
    FileNotFoundError
        If dataset directory cannot be found or is invalid
    """
    if project_root is None:
        project_root = find_project_root()

    # search order for dataset directory
    search_paths = []

    # 1. Kaggle input directory
    if is_kaggle():
        search_paths.append(Path("/kaggle/input/g2net-gravitational-wave-detection"))

    # 2. Environment variable
    env_path = os.getenv("G2NET_DATASET_PATH")
    if env_path:
        search_paths.append(Path(env_path))

    # 3. External drive (local development)
    search_paths.append(Path("D:/Programming/g2net-gravitational-wave-detection"))

    # 4. Full dataset under project data directory
    search_paths.append(project_root / "data" / "g2net-gravitational-wave-detection")

    # 5. Exploration sample (1000 samples, ~100 MB)
    search_paths.append(sample_dataset_dir(project_root))

    # find first valid path
    for candidate_path in search_paths:
        if _is_valid_dataset(candidate_path):
            return candidate_path

    # nothing found
    error_msg = (
        "G2Net dataset not found. Searched:\n" +
        "\n".join(f"  - {p}" for p in search_paths) +
        "\n\nOptions:\n"
        "1. Set G2NET_DATASET_PATH to your local copy of the full dataset\n"
        "2. Call download_sample_dataset() to fetch a 1000-sample subset (~100 MB)\n"
        "3. Run this notebook on Kaggle with the competition dataset attached"
    )
    raise FileNotFoundError(error_msg)


# ---------------------------------------------------------------------
#                       labels and samples
# ---------------------------------------------------------------------

def load_labels(dataset_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Read the labels file (training_labels.csv) from the dataset directory.

    Parameters
    ----------
    dataset_dir : Path, optional
        Dataset directory. If None, it is resolved via find_dataset_dir().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'id' (str), 'target' (int).

    Raises
    ------
    FileNotFoundError
        If the labels file does not exist.
    """
    if dataset_dir is None:
        dataset_dir = find_dataset_dir()

    labels_path = dataset_dir / "training_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    df = pd.read_csv(labels_path, dtype={"id": str})
    return df

def sample_path(sample_id: str, split_dir: Path) -> Path:
    """
    Build the path to a .npy sample for the G2Net dataset.

    Layout:split/<id[0]>/<id[1]>/<id[2]>/<id>.npy
    """
    sample_id = str(sample_id).strip()      # just in case

    if len(sample_id) < 3:
        raise ValueError(
            f"sample_id too short for G2Net nested layout: {sample_id!r} (len={len(sample_id)})"
        )

    return (
        split_dir
        / sample_id[0]
        / sample_id[1]
        / sample_id[2]
        / f"{sample_id}.npy"
    )

def load_sample(sample_id: str, split: str = "train", dataset_dir: Optional[Path] = None) -> np.ndarray:
    """
    Load a waveform sample from the G2Net dataset.
    """
    if dataset_dir is None:
        dataset_dir = find_dataset_dir()

    split = str(split).strip().lower()
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split!r}. Expected 'train' or 'test'.")

    split_dir = dataset_dir / split

    try:
        p = sample_path(sample_id, split_dir)
    except ValueError as e:
        raise ValueError(f"Invalid sample_id {sample_id!r}: {e}") from e

    if not p.is_file():
        raise FileNotFoundError(f"Sample not found for id={sample_id}. Expected: {p}")

    return np.load(p)