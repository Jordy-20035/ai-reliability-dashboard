import { api } from './client'
import type {
  ApiListResponse,
  DatasetVersion,
  LifecycleExperiment,
  LifecycleModel,
  InferenceResponse,
  OverviewResponse,
  ProvenanceRow,
  RetrainScenario,
  RunRecord,
  Scenario,
} from '../types'

export async function getOverview() {
  const res = await api.get<OverviewResponse>('/api/overview')
  return res.data
}

export async function getRuns(limit = 50) {
  const res = await api.get<ApiListResponse<RunRecord>>('/api/orchestration/runs', {
    params: { limit },
  })
  return res.data
}

export type DriftCheckOptions = {
  currentCsvPath?: string
  fraudD1Path?: string
  fraudD2Path?: string
  fraudD3Path?: string
}

export async function runDriftCheck(scenario: Scenario, opts?: DriftCheckOptions) {
  const params: Record<string, string> = { scenario }
  if (scenario === 'incoming_csv' && opts?.currentCsvPath?.trim()) {
    params.current_csv_path = opts.currentCsvPath.trim()
  }
  if (opts?.fraudD1Path?.trim()) params.fraud_d1_path = opts.fraudD1Path.trim()
  if (opts?.fraudD2Path?.trim()) params.fraud_d2_path = opts.fraudD2Path.trim()
  if (opts?.fraudD3Path?.trim()) params.fraud_d3_path = opts.fraudD3Path.trim()
  const res = await api.post('/api/orchestration/check-once', null, {
    params,
  })
  return res.data
}

export type RetrainRunOptions = { fraudD1Path?: string; fraudD2Path?: string }

export async function runRetrain(scenario: RetrainScenario, opts?: RetrainRunOptions) {
  const body: Record<string, unknown> = { scenario }
  if (scenario === 'fraud_retrain_d1_d2') {
    if (opts?.fraudD1Path?.trim()) body.fraud_d1_path = opts.fraudD1Path.trim()
    if (opts?.fraudD2Path?.trim()) body.fraud_d2_path = opts.fraudD2Path.trim()
  }
  const res = await api.post('/api/retraining/run', body)
  return res.data
}

export async function getModels() {
  const res = await api.get<ApiListResponse<LifecycleModel>>('/api/lifecycle/models')
  return res.data
}

export async function getExperiments(limit = 100) {
  const res = await api.get<ApiListResponse<LifecycleExperiment>>('/api/lifecycle/experiments', {
    params: { limit },
  })
  return res.data
}

export async function getProductionPointer() {
  const res = await api.get<{ production_model_row_id: number | null }>('/api/lifecycle/production')
  return res.data
}

export async function promoteModel(lifecycleModelId: number, toStage: string) {
  const res = await api.post('/api/lifecycle/promote', {
    lifecycle_model_id: lifecycleModelId,
    to_stage: toStage,
  })
  return res.data
}

export async function getDatasets(limit = 100) {
  const res = await api.get<ApiListResponse<DatasetVersion>>('/api/data/datasets', {
    params: { limit },
  })
  return res.data
}

export async function getProvenance(limit = 100) {
  const res = await api.get<ApiListResponse<ProvenanceRow>>('/api/data/provenance', {
    params: { limit },
  })
  return res.data
}

export async function predictProduction(
  rows: Array<Record<string, unknown>>,
  profile: 'adult' | 'fraud' = 'adult',
) {
  const res = await api.post<InferenceResponse>('/api/inference/predict', { rows, profile })
  return res.data
}

