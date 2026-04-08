export type Scenario = 'random_holdout' | 'age_shift'

export interface ApiListResponse<T> {
  items: T[]
  count: number
}

export interface RunRecord {
  id: number
  started_at: string
  finished_at: string
  scenario: string
  policy_triggered: boolean
  trigger_reasons: string[]
  summary: Record<string, number>
}

export interface LifecycleModel {
  id: number
  version_num: number
  experiment_id: number
  artifact_path: string
  stage: string
  created_at: string
  metrics: Record<string, number>
  notes: string | null
}

export interface LifecycleExperiment {
  id: number
  name: string
  created_at: string
  params: Record<string, unknown>
  metrics: Record<string, number>
  scenario: string | null
  notes: string | null
  git_sha: string | null
}

export interface DatasetVersion {
  id: number
  name: string
  kind: string
  source_path: string | null
  content_hash: string
  row_count: number
  columns: string[]
  created_at: string
  notes: string | null
}

export interface ProvenanceRow {
  id: number
  lifecycle_experiment_id: number | null
  lifecycle_model_version_num: number | null
  dataset_version_id: number
  baseline_snapshot_id: number | null
  git_sha: string | null
  extra: Record<string, unknown>
  created_at: string
}

export interface OverviewResponse {
  kpis: {
    n_runs: number
    n_models: number
    n_experiments: number
    n_datasets: number
    n_baselines: number
    production_model_row_id: number | null
    last_run_policy_triggered: boolean | null
  }
  last_run: RunRecord | null
  latest_model: LifecycleModel | null
  latest_experiment: LifecycleExperiment | null
}

