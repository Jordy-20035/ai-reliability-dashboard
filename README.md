# Trustworthy AI — drift + orchestration

## 1. Data drift detection

Statistical drift vs a **training baseline**: **PSI** (binned), **Kolmogorov–Smirnov** (numeric), **Chi-square** (categorical homogeneity).

- Place `adult.csv` in `data/raw/` (see `data/raw/README.txt`).
- **Baseline**: `BaselineProfile.fit(reference_df, ...)` freezes PSI bin edges from the reference sample.
- **Score**: `run_drift_analysis(reference, current, baseline)` compares batches (same schema).

```bash
pip install -r requirements.txt
python -m src.drift_detection
```

## 2. Automation / orchestration (“brain”)

Runs drift checks **on a schedule** or **on demand** (HTTP), evaluates **threshold policies**, runs **actions** (log + placeholder retrain), and appends runs to **SQLite** (`artifacts/orchestration.db`).

```bash
python -m src.orchestration init-baseline
python -m src.orchestration check-once --scenario random_holdout
python -m src.orchestration check-once --scenario age_shift
python -m src.orchestration history
# python -m src.orchestration serve --interval 60
# python -m src.orchestration serve-http --port 8000
```

```bash
pytest
```
