"""
Download G2Net dataset from Kaggle.

Requires kaggle API credentials configured (~/.kaggle/kaggle.json on Unix,
or %USERPROFILE%\.kaggle\kaggle.json on Windows).

To set up Kaggle API:
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" - this downloads kaggle.json
3. Place kaggle.json in the appropriate directory
4. Install kaggle: pip install kaggle
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import zipfile


def download_g2net_dataset(output_dir: Path | str | None = None, unzip: bool = True) -> Path:
    """
    Download the G2Net gravitational wave detection dataset from Kaggle.

    Parameters
    ----------
    output_dir : Path | str | None
        Directory to download the dataset to.
        If None, downloads to project_root/data/g2net-gravitational-wave-detection
    unzip : bool
        Whether to automatically unzip the downloaded files (default: True)

    Returns
    -------
    Path
        Path to the downloaded dataset directory

    Raises
    ------
    ImportError
        If kaggle package is not installed
    RuntimeError
        If kaggle API is not configured or download fails
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle package not installed. Install with: pip install kaggle"
        )

    # Determine output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "g2net-gravitational-wave-detection"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading G2Net dataset to: {output_dir}")
    print("This may take a while (dataset is ~120GB)...")

    # Initialize Kaggle API
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        raise RuntimeError(
            f"Failed to authenticate with Kaggle API. "
            f"Make sure kaggle.json is in ~/.kaggle/ (or %USERPROFILE%\\.kaggle\\ on Windows). "
            f"Error: {e}"
        )

    # Download competition files
    competition_name = "g2net-gravitational-wave-detection"

    try:
        api.competition_download_files(
            competition_name,
            path=str(output_dir),
            quiet=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")

    # Unzip if requested
    if unzip:
        print("\nUnzipping downloaded files...")
        zip_files = list(output_dir.glob("*.zip"))

        for zip_file in zip_files:
            print(f"Extracting {zip_file.name}...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)

                # Optionally remove zip file after extraction
                # zip_file.unlink()
                print(f"Extracted {zip_file.name}")
            except Exception as e:
                print(f"Warning: Failed to extract {zip_file.name}: {e}")

    print(f"\nDataset downloaded successfully to: {output_dir}")
    return output_dir


def main():
    """CLI entry point for downloading the dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download G2Net gravitational wave detection dataset from Kaggle"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the dataset (default: project_root/data/g2net-gravitational-wave-detection)"
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Don't automatically unzip downloaded files"
    )

    args = parser.parse_args()

    try:
        download_g2net_dataset(
            output_dir=args.output_dir,
            unzip=not args.no_unzip
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
