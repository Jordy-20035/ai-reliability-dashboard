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

Runs drift checks **on a schedule** or **on demand** (HTTP), evaluates **threshold policies**, runs **actions** (log + **automated retrain** when policy fires), and appends runs to **SQLite** (`artifacts/orchestration.db`).

```bash
python -m src.orchestration init-baseline
python -m src.orchestration check-once --scenario random_holdout
python -m src.orchestration check-once --scenario age_shift
python -m src.orchestration history
# python -m src.orchestration serve --interval 60
# python -m src.orchestration serve-http --port 8000
```

## 3. Automated retraining pipeline

When drift policy triggers, **`RetrainPipelineAction`** merges **labeled** reference + current rows, trains a **HistGradientBoosting** model (sklearn pipeline with one-hot categoricals), evaluates on a holdout split, saves **`artifacts/models/model_vN.joblib`**, updates **`registry.json`**, and **promotes** the new model to **`champion.json`** if the primary metric (default **macro F1**) is at least as good as the current champion. Each run is also **recorded in the lifecycle DB** (experiments + model rows).

Manual run (same data split as orchestration):

```bash
python -m src.retraining --scenario random_holdout
```

## 4. Model lifecycle management

**SQLite** (`artifacts/lifecycle.db`): **experiments** (params, metrics, scenario, optional **git SHA**), **model versions** (artifact path, metrics, **deployment stage**), and a **production** pointer. Stages: `development` → `staging` → `production`; superseded production models move to `archived`.

After each retrain, **`sync_from_retrain`** inserts an experiment + model row (development, or production if metric promotion applies). Manual moves:

```bash
python -m src.lifecycle list-experiments
python -m src.lifecycle list-models
python -m src.lifecycle list-models --stage production
python -m src.lifecycle promote <model_row_id> --to staging
python -m src.lifecycle production-id
```

```bash
pytest
```
