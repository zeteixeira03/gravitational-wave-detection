# Data Pipeline

## Preprocessing and Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONE-TIME LOCAL PREPROCESSING                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw G2Net Data (70GB)          create_tensors.py                          │
│   <path-to-dataset>/       ──────────────────────────►  Tensor shards      │
│                                                          <output-path>/     │
│   For each sample:                                       g2net-preprocessed/ │
│   1. Load .npy file                                      ├── shard_00.pt    │
│   2. Bandpass filter (20-500 Hz)                         ├── shard_01.pt    │
│   3. Whiten using avg_psd                                ├── ...            │
│   4. Tukey window                                        ├── avg_psd.npz    │
│   5. Normalize                                           └── metadata.json  │
│   6. Serialize to .pt shard (torch.save)                                    │
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
│       └── shard_*.pt                       (preprocessed signals)           │
│                                                                             │
│   kaggle/train.py                                                           │
│        │                                                                    │
│        ▼                                                                    │
│   main()  ───────►  train_from_tensors()  ───────►  DIYModel               │
│        │                       │                           │                │
│        │                       │ GWTensorDataset           │ Deep residual  │
│        │                       │ + DataLoader              │ 1D CNN (10     │
│        │                       │ (shard streaming)         │ res blocks,    │
│        │                       │                           │ stochastic     │
│        │                       │                           │ depth, GeM)    │
│        │                       ▼                           │ 3-layer head   │
│        │                    fit()                          ▼                │
│        │                       │                      Trained Weights       │
│        │                       │ - Early stopping                           │
│        │                       │ - LR scheduling                            │
│        │                       │ - Best weights restore                     │
│        │                       ▼                                            │
│        │              generate_plots()                                      │
│        │                       │                                            │
│        ▼                       ▼                                            │
│   /kaggle/working/models/saved/                                             │
│   ├── diy_YYYYMMDD_HHMM_weights.pt                                          │
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
| Preprocessed tensor shards | ~2.4 GB/shard |
| avg_psd.npz | ~45 KB |
| Model weights | ~6 MB |
| src/ code (zipped) | ~150 KB |
