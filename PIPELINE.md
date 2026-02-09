# Data Pipeline

## Preprocessing and Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONE-TIME LOCAL PREPROCESSING                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw G2Net Data (70GB)          create_tfrecords.py                        │
│   D:/Programming/g2net-...  ──────────────────────────►  TFRecords (18GB)   │
│                                                          D:/Programming/     │
│   For each sample:                                       g2net-preprocessed/ │
│   1. Load .npy file                                      ├── train.tfrecord  │
│   2. Bandpass filter (20-500 Hz)                         ├── avg_psd.npz     │
│   3. Whiten using avg_psd                                └── metadata.json   │
│   4. Tukey window                                                            │
│   5. Normalize                                                               │
│   6. Serialize to TFRecord                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Upload once to Kaggle
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KAGGLE TRAINING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Kaggle Input Datasets:                                                    │
│   ├── /kaggle/input/gw-src-code/           (your src/ code)                 │
│   └── /kaggle/input/g2net-preprocessed-tfrecords/                           │
│       └── train.tfrecord                   (preprocessed signals)           │
│                                                                             │
│   kaggle/train.py                                                           │
│        │                                                                    │
│        ▼                                                                    │
│   main()  ───────►  train_from_tfrecords()  ───────►  DIYModel              │
│        │                       │                           │                │
│        │                       │ tf.data.TFRecordDataset   │ 1D CNN         │
│        │                       │ (fast sequential read)    │ 4 conv blocks  │
│        │                       ▼                           │ GeM pooling    │
│        │                    fit()                          │ 3 FC layers    │
│        │                       │                           ▼                │
│        │                       │ - Early stopping          Trained Weights  │
│        │                       │ - LR scheduling                            │
│        │                       │ - Best weights restore                     │
│        │                       ▼                                            │
│        │              generate_plots()                                      │
│        │                       │                                            │
│        ▼                       ▼                                            │
│   /kaggle/working/models/saved/                                             │
│   ├── diy_YYYYMMDD_HHMM_weights.npz                                         │
│   ├── diy_YYYYMMDD_HHMM_config.json                                         │
│   ├── diy_YYYYMMDD_HHMM_metrics.json                                        │
│   └── plots/                                                                │
│       ├── diy_YYYYMMDD_HHMM_dashboard.png                                   │
│       ├── diy_YYYYMMDD_HHMM_learning_curves.png                             │
│       ├── diy_YYYYMMDD_HHMM_roc_curve.png                                   │
│       └── ...                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Sizes

| Data | Size |
|------|------|
| Raw G2Net dataset | ~70 GB |
| Preprocessed TFRecords | ~18 GB |
| avg_psd.npz | ~45 KB |
| Model weights | ~4 MB |
| src/ code (zipped) | ~150 KB |
