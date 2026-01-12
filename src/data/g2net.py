from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project / dataset discovery
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


def find_dataset_dir(project_root: Optional[Path] = None) -> Path:
    """
    Find the G2Net dataset directory.

    Search order:
    1. Environment variable G2NET_DATASET_PATH
    2. project_root/data/g2net-gravitational-wave-detection
    3. Current directory / g2net-gravitational-wave-detection

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

    # Search order for dataset directory
    search_paths = []

    # 1. Environment variable (highest priority)
    env_path = os.getenv("G2NET_DATASET_PATH")
    if env_path:
        search_paths.append(Path(env_path))

    # 2. Project data directory
    search_paths.append(project_root / "data" / "g2net-gravitational-wave-detection")

    # 3. Current directory
    search_paths.append(Path.cwd() / "g2net-gravitational-wave-detection")

    # 4. Legacy hardcoded path
    legacy_path = Path(r"D:\Programming\g2net-gravitational-wave-detection")
    if legacy_path.exists():
        search_paths.append(legacy_path)

    # Find first valid path (must contain a 'train' file)
    for candidate_path in search_paths:
        if candidate_path.exists() and (candidate_path / "train").is_dir():
            return candidate_path

    # If no valid path found, provide helpful error message
    error_msg = (
        "G2Net dataset directory not found. Tried the following locations:\n" +
        "\n".join(f"  - {p}" for p in search_paths) +
        "\n\nTo fix this, either:\n"
        "1. Set environment variable: G2NET_DATASET_PATH=/path/to/dataset\n"
        "2. Download dataset using: python src/data/download_data.py\n"
        "3. Manually place dataset in: " + str(project_root / "data" / "g2net-gravitational-wave-detection")
    )
    raise FileNotFoundError(error_msg)


# ---------------------------------------------------------------------
# Labels and samples
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

    Expected layout (always):
        split/<id[0]>/<id[1]>/<id[2]>/<id>.npy
    """
    sample_id = str(sample_id).strip()

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


def load_sample(
    sample_id: str,
    split: str = "train",
    dataset_dir: Optional[Path] = None,
) -> np.ndarray:
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

