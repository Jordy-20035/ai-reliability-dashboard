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
python -m src.orchestration check-once --scenario incoming_csv --current-csv-path ./data/raw/adult.csv
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

## 5. Data management (SQLite)

**Database:** `artifacts/data_management.db` (gitignored with other `artifacts/`).

| Table | Purpose |
|-------|---------|
| **`dataset_versions`** | Content **SHA-256** fingerprint, row count, column list, name/kind; deduped by hash (same bytes → same id). |
| **`baseline_snapshots`** | Drift **`baseline_profile.json`** path + hash; optional link to a reference dataset row + **JSON distribution summary** (quantiles / top categories). |
| **`training_provenance`** | Links a retrain to **dataset_version_id**, optional **baseline_snapshot_id**, **lifecycle experiment id**, model version number, **git SHA**, extra JSON. |

**Automatic hooks**

- **`python -m src.orchestration init-baseline`** — registers the **reference feature** snapshot used for PSI bins and a **baseline snapshot** with a compact **distribution summary**.
- **Each `run_retrain_pipeline`** — registers the **merged training** dataframe fingerprint and a provenance row (baseline snapshot if `artifacts/baseline_profile.json` exists).

**CLI**

```bash
python -m src.data_management register-raw
python -m src.data_management list-datasets
python -m src.data_management list-baselines
python -m src.data_management list-provenance
```

## 6. Unified API (Phase 1)

Run:

```bash
python -m src.api --host 127.0.0.1 --port 8000
```

Key endpoints:

- `GET /health`
- `GET /api/overview`
- `GET /api/orchestration/runs?limit=50`
- `GET /api/ops/stats`
- `POST /api/orchestration/check-once?scenario=random_holdout`
- `POST /api/orchestration/check-once?scenario=incoming_csv&current_csv_path=./data/raw/adult.csv`
- `POST /api/retraining/run` body: `{"scenario":"random_holdout"}` or `{"scenario":"age_shift"}`
- `POST /api/inference/predict` body: `{"rows":[{...adult feature columns...}]}`
- `GET /api/lifecycle/models`
- `GET /api/lifecycle/experiments`
- `GET /api/lifecycle/production`
- `POST /api/lifecycle/promote` body: `{"lifecycle_model_id":1,"to_stage":"staging"}`
- `GET /api/data/datasets`
- `GET /api/data/baselines`
- `GET /api/data/provenance`

## 7. Custom dashboard (Phase 2, React)

Frontend lives in `frontend/` and consumes the API-first backend (`/api/*`).

```bash
# Terminal A
python -m src.api --host 127.0.0.1 --port 8000

# Terminal B
cd frontend
npm install
npm run dev
```

Optional API URL override (default is `http://127.0.0.1:8000`):

```bash
# in frontend/.env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

```bash
pytest
```

## 8. Phase 3 operations (deployment + automation + alerts)

### A) Env-driven orchestration controls

The orchestration layer now supports operational env vars:

- `ORCH_ALERT_WEBHOOK_URL` — optional webhook called when policy triggers.
- `ORCH_ENABLE_AUTO_RETRAIN` — `true/false`; disable side-effect retraining for monitor-only mode.
- `ORCH_SCHEDULER_INTERVAL` — scheduler interval seconds.
- `ORCH_CURRENT_CSV_PATH` — incoming batch CSV path for `scenario=incoming_csv`.

CLI also accepts:

```bash
python -m src.orchestration check-once --disable-auto-retrain --alert-webhook-url https://example.test/webhook
python -m src.orchestration serve --interval 0
```

(`--interval 0` means: use `ORCH_SCHEDULER_INTERVAL` from env/config.)

### B) Operational stats endpoint

```bash
GET /api/ops/stats
```

Response includes total runs, triggered runs, ok runs, and trigger rate.

### C) Docker Compose stack

Files:

- `Dockerfile` (backend API + scheduler image)
- `frontend/Dockerfile` + `frontend/nginx.conf` (SPA build + serve)
- `docker-compose.yml` (api + scheduler + frontend)
- `.env.example` (Phase 3 env knobs)

Run:

```bash
# create .env from .env.example, then:
docker compose up --build
```

Then:

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8090`
