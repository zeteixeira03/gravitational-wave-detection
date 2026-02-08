"""
Create TFRecord dataset with preprocessed (whitened) signals.

Run this locally once to generate TFRecords, then upload to Kaggle as a dataset.

Usage:
    python src/data/create_tfrecords.py --input /path/to/g2net-dataset --output /path/to/output

The output directory will contain:
    - train.tfrecord (preprocessed signals + labels)
    - metadata.json (sample count, preprocessing info)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.g2net import load_labels, load_sample
from data.preprocessing import preprocess_sample, load_psd
from data.compute_psd import compute_and_save_average_psd


# ============================================================================
#                              TFRECORD UTILS
# ============================================================================

def _bytes_feature(value):
    """Returns a bytes_list from a numpy array."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def _int64_feature(value):
    """Returns an int64_list from an int."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_sample(signal: np.ndarray, label: int) -> bytes:
    """Serialize a preprocessed signal and label to TFRecord format."""
    feature = {
        'signal': _bytes_feature(signal.astype(np.float32)),
        'label': _int64_feature(label),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# ============================================================================
#                              MAIN
# ============================================================================

def create_tfrecords(input_dir: Path, output_dir: Path, psd_path: Path = None):
    """
    Create TFRecord dataset from raw G2Net data.

    Parameters
    ----------
    input_dir
        Path to g2net-gravitational-wave-detection directory (contains train/, training_labels.csv)
    output_dir
        Path to save TFRecords
    psd_path
        Path to precomputed PSD file. If None, computes from noise samples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # load or compute PSD
    if psd_path and psd_path.exists():
        print(f"Loading PSD from: {psd_path}")
        avg_psd = load_psd(psd_path)
    else:
        print("Computing average PSD from noise samples...")
        psd_path = output_dir / "avg_psd.npz"
        avg_psd = compute_and_save_average_psd(
            n_samples=10000, output_path=psd_path, dataset_dir=input_dir
        )

    # load labels
    print(f"\nLoading labels from: {input_dir}")
    df = load_labels(input_dir)
    n_samples = len(df)
    print(f"Total samples: {n_samples}")

    # create TFRecord writer
    tfrecord_path = output_dir / "train.tfrecord"
    print(f"\nWriting TFRecords to: {tfrecord_path}")

    written = 0
    failed = 0

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for _, row in tqdm(df.iterrows(), total=n_samples, desc="Processing"):
            sample_id = str(row['id'])
            label = int(row['target'])

            try:
                # load raw signal
                raw = load_sample(sample_id, dataset_dir=input_dir)

                # preprocess (bandpass, whiten, window, normalize)
                processed = preprocess_sample(raw, avg_psd)

                # skip if NaN/Inf
                if np.isnan(processed).any() or np.isinf(processed).any():
                    failed += 1
                    continue

                # serialize and write
                serialized = serialize_sample(processed, label)
                writer.write(serialized)
                written += 1

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"\nError processing {sample_id}: {e}")

    # save metadata
    metadata = {
        'n_samples': written,
        'n_failed': failed,
        'signal_shape': [3, 4096],
        'dtype': 'float32',
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  Written: {written}")
    print(f"  Failed: {failed}")
    print(f"  Output: {tfrecord_path}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TFRecord dataset from G2Net data")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to g2net-gravitational-wave-detection directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save TFRecords")
    parser.add_argument("--psd", type=str, default=None,
                        help="Path to precomputed avg_psd.npz (optional)")

    args = parser.parse_args()

    create_tfrecords(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        psd_path=Path(args.psd) if args.psd else None,
    )
