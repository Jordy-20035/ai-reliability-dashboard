# Data drift detection

Statistical drift vs a **training baseline**: **PSI** (binned), **Kolmogorov–Smirnov** (numeric), **Chi-square** (categorical homogeneity).

- Place `adult.csv` in `data/raw/` (see `data/raw/README.txt`).
- **Baseline**: `BaselineProfile.fit(reference_df, ...)` freezes PSI bin edges from the reference sample.
- **Score**: `run_drift_analysis(reference, current, baseline)` compares batches (same schema).

```bash
pip install -r requirements.txt
python -m src.drift_detection
pytest
```
