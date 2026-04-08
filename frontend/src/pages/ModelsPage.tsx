import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { EmptyGridOverlay } from '../components/EmptyGridOverlay'
import { getExperiments, getModels, getProductionPointer, promoteModel, runRetrain } from '../api/endpoints'
import type { LifecycleExperiment, LifecycleModel } from '../types'
import { getErrorMessage } from '../utils/errors'
type RetrainScenario = 'random_holdout' | 'age_shift'

const modelCols: GridColDef<LifecycleModel>[] = [
  { field: 'id', headerName: 'Row ID', width: 90 },
  { field: 'version_num', headerName: 'Version', width: 100 },
  { field: 'stage', headerName: 'Stage', width: 130 },
  { field: 'created_at', headerName: 'Created', width: 220 },
  { field: 'experiment_id', headerName: 'Exp ID', width: 100 },
  {
    field: 'f1',
    headerName: 'F1 Macro',
    width: 110,
    valueGetter: (_, row) => row.metrics?.f1_macro ?? '-',
  },
  { field: 'notes', headerName: 'Notes', flex: 1, minWidth: 220 },
]

const expCols: GridColDef<LifecycleExperiment>[] = [
  { field: 'id', headerName: 'ID', width: 80 },
  { field: 'name', headerName: 'Name', minWidth: 240, flex: 1 },
  { field: 'scenario', headerName: 'Scenario', width: 130 },
  { field: 'created_at', headerName: 'Created', width: 220 },
  { field: 'git_sha', headerName: 'Git SHA', width: 160 },
]

export function ModelsPage() {
  const [models, setModels] = useState<LifecycleModel[]>([])
  const [experiments, setExperiments] = useState<LifecycleExperiment[]>([])
  const [productionId, setProductionId] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [scenario, setScenario] = useState<RetrainScenario>('random_holdout')
  const [promoteId, setPromoteId] = useState('')
  const [promoteStage, setPromoteStage] = useState('staging')
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const [m, e, p] = await Promise.all([getModels(), getExperiments(100), getProductionPointer()])
      setModels(m.items)
      setExperiments(e.items)
      setProductionId(p.production_model_row_id)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  async function onRetrain() {
    setMessage(null)
    setLoading(true)
    try {
      const res = await runRetrain(scenario)
      setMessage(`Retrain done: v${String(res.version)} promoted=${String(res.promoted)}`)
      await load()
    } catch (e) {
      setMessage(`Retrain failed: ${getErrorMessage(e)}`)
      setLoading(false)
    }
  }

  async function onPromote() {
    const id = Number(promoteId)
    if (!Number.isFinite(id)) {
      setMessage('Provide a numeric lifecycle model row id.')
      return
    }
    setLoading(true)
    setMessage(null)
    try {
      await promoteModel(id, promoteStage)
      setMessage(`Model row ${String(id)} moved to ${promoteStage}.`)
      await load()
    } catch (e) {
      setMessage(`Promotion failed: ${getErrorMessage(e)}`)
      setLoading(false)
    }
  }

  return (
    <Stack spacing={2}>
      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error">
          Could not load models — start API: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        Model Lifecycle & Control
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 960 }}>
        <strong>Trigger Retraining</strong> trains a new version and registers it in the lifecycle store. Then
        pick a <strong>Row ID</strong> from the Model Versions table and <strong>Promote stage</strong> to move
        it toward production (or archive). Production pointer:{' '}
        <strong>{productionId ?? 'not set'}</strong>.
      </Typography>
      {message && <Alert severity="info">{message}</Alert>}

      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
        <Select
          size="small"
          value={scenario}
          onChange={(e) => setScenario(e.target.value as RetrainScenario)}
        >
          <MenuItem value="random_holdout">random_holdout</MenuItem>
          <MenuItem value="age_shift">age_shift</MenuItem>
        </Select>
        <Button
          variant="contained"
          onClick={() => void onRetrain()}
          disabled={loading}
          aria-label="Trigger model retraining for selected scenario"
        >
          Trigger Retraining
        </Button>
        <TextField
          size="small"
          label="Lifecycle Row ID"
          value={promoteId}
          onChange={(e) => setPromoteId(e.target.value)}
        />
        <Select size="small" value={promoteStage} onChange={(e) => setPromoteStage(e.target.value)}>
          <MenuItem value="staging">staging</MenuItem>
          <MenuItem value="production">production</MenuItem>
          <MenuItem value="archived">archived</MenuItem>
        </Select>
        <Button
          variant="outlined"
          onClick={() => void onPromote()}
          disabled={loading}
          aria-label="Promote lifecycle model row to selected stage"
        >
          Promote Stage
        </Button>
      </Box>

      <Paper sx={{ p: 1 }}>
        <Typography variant="h6" sx={{ p: 1 }}>
          Model Versions
        </Typography>
        <DataGrid
          rows={models}
          columns={modelCols}
          getRowId={(r) => r.id}
          loading={loading}
          autoHeight
          pageSizeOptions={[10, 20, 50]}
          initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
          slots={{
            noRowsOverlay: () => (
              <EmptyGridOverlay message="No registered models yet. Run Trigger Retraining to create a version, or seed data via the training CLI/API." />
            ),
          }}
        />
      </Paper>

      <Paper sx={{ p: 1 }}>
        <Typography variant="h6" sx={{ p: 1 }}>
          Experiments
        </Typography>
        <DataGrid
          rows={experiments}
          columns={expCols}
          getRowId={(r) => r.id}
          loading={loading}
          autoHeight
          pageSizeOptions={[10, 20, 50]}
          initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
          slots={{
            noRowsOverlay: () => (
              <EmptyGridOverlay message="No experiments yet. They appear when training runs register lineage (e.g. after retraining)." />
            ),
          }}
        />
      </Paper>
    </Stack>
  )
}

