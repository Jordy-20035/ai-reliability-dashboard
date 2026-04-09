import { api } from './client'
import type {
  ApiListResponse,
  DatasetVersion,
  LifecycleExperiment,
  LifecycleModel,
  InferenceResponse,
  OverviewResponse,
  ProvenanceRow,
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

export async function runDriftCheck(scenario: Scenario, currentCsvPath?: string) {
  const params: Record<string, string> = { scenario }
  if (scenario === 'incoming_csv' && currentCsvPath?.trim()) {
    params.current_csv_path = currentCsvPath.trim()
  }
  const res = await api.post('/api/orchestration/check-once', null, {
    params,
  })
  return res.data
}

export async function runRetrain(scenario: 'random_holdout' | 'age_shift') {
  const res = await api.post('/api/retraining/run', { scenario })
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

export async function predictProduction(rows: Array<Record<string, unknown>>) {
  const res = await api.post<InferenceResponse>('/api/inference/predict', { rows })
  return res.data
}

