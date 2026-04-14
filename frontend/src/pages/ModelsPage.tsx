import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Divider,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { Publish, Refresh } from '@mui/icons-material'
import { EmptyGridOverlay } from '../components/EmptyGridOverlay'
import { StageBadge } from '../components/StageBadge'
import { getExperiments, getModels, getProductionPointer, promoteModel, runRetrain } from '../api/endpoints'
import type { LifecycleExperiment, LifecycleModel, RetrainScenario } from '../types'
import { getErrorMessage } from '../utils/errors'

const modelCols: GridColDef<LifecycleModel>[] = [
  { field: 'id', headerName: 'Row ID', width: 80 },
  { field: 'version_num', headerName: 'Version', width: 90 },
  {
    field: 'stage',
    headerName: 'Stage',
    width: 130,
    renderCell: (params) => <StageBadge stage={params.value as string} />,
  },
  { field: 'created_at', headerName: 'Created', width: 190 },
  { field: 'experiment_id', headerName: 'Exp ID', width: 80 },
  {
    field: 'f1',
    headerName: 'F1 Macro',
    width: 100,
    valueGetter: (_, row) => row.metrics?.f1_macro ?? '-',
  },
  {
    field: 'accuracy',
    headerName: 'Accuracy',
    width: 100,
    valueGetter: (_, row) => row.metrics?.accuracy != null ? (row.metrics.accuracy as number).toFixed(4) : '-',
  },
  { field: 'notes', headerName: 'Notes', flex: 1, minWidth: 200 },
]

const expCols: GridColDef<LifecycleExperiment>[] = [
  { field: 'id', headerName: 'ID', width: 70 },
  { field: 'name', headerName: 'Name', minWidth: 240, flex: 1 },
  { field: 'scenario', headerName: 'Scenario', width: 140 },
  { field: 'created_at', headerName: 'Created', width: 190 },
  { field: 'git_sha', headerName: 'Git SHA', width: 140 },
]

export function ModelsPage() {
  const [models, setModels] = useState<LifecycleModel[]>([])
  const [experiments, setExperiments] = useState<LifecycleExperiment[]>([])
  const [productionId, setProductionId] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [scenario, setScenario] = useState<RetrainScenario>('random_holdout')
  const [fraudD1Path, setFraudD1Path] = useState('')
  const [fraudD2Path, setFraudD2Path] = useState('')
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
      const res = await runRetrain(scenario, { fraudD1Path, fraudD2Path })
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
    <Stack spacing={2.5}>
      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error">
          Could not load models — start API: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}
      {message && <Alert severity="info">{message}</Alert>}

      <Typography variant="body2" color="text.secondary">
        Production pointer: <strong>{productionId ?? 'not set'}</strong>
      </Typography>

      {/* ---- Action panels ---- */}
      <Paper variant="outlined" sx={{ p: 2.5 }}>
        <Typography variant="h6" gutterBottom>
          Trigger Retraining
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
          <Select size="small" value={scenario} onChange={(e) => setScenario(e.target.value as RetrainScenario)}>
            <MenuItem value="random_holdout">random_holdout</MenuItem>
            <MenuItem value="age_shift">age_shift</MenuItem>
            <MenuItem value="fraud_retrain_d1_d2">fraud_retrain_d1_d2</MenuItem>
          </Select>
          {scenario === 'fraud_retrain_d1_d2' && (
            <>
              <TextField size="small" label="D1 (opt)" value={fraudD1Path} onChange={(e) => setFraudD1Path(e.target.value)} sx={{ minWidth: 180 }} />
              <TextField size="small" label="D2 (opt)" value={fraudD2Path} onChange={(e) => setFraudD2Path(e.target.value)} sx={{ minWidth: 180 }} />
            </>
          )}
          <Button variant="contained" startIcon={<Refresh />} onClick={() => void onRetrain()} disabled={loading}>
            Retrain
          </Button>
        </Box>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2.5 }}>
        <Typography variant="h6" gutterBottom>
          Promote Stage
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField size="small" label="Row ID" value={promoteId} onChange={(e) => setPromoteId(e.target.value)} sx={{ width: 110 }} />
          <Select size="small" value={promoteStage} onChange={(e) => setPromoteStage(e.target.value)}>
            <MenuItem value="staging">staging</MenuItem>
            <MenuItem value="production">production</MenuItem>
            <MenuItem value="archived">archived</MenuItem>
          </Select>
          <Button variant="outlined" startIcon={<Publish />} onClick={() => void onPromote()} disabled={loading}>
            Promote
          </Button>
        </Box>
      </Paper>

      {/* ---- Tables ---- */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ px: 1, pb: 1 }}>
          Model Registry &amp; Lifecycle
        </Typography>
        <DataGrid
          rows={models}
          columns={modelCols}
          getRowId={(r) => r.id}
          loading={loading}
          autoHeight
          pageSizeOptions={[10, 20, 50]}
          initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
          sx={{
            border: 'none',
            '& .MuiDataGrid-columnHeaders': { bgcolor: '#f8fafc' },
            '& .MuiDataGrid-row:hover': { bgcolor: '#fafbff' },
          }}
          slots={{
            noRowsOverlay: () => (
              <EmptyGridOverlay message="No registered models yet. Run Trigger Retraining to create a version." />
            ),
          }}
        />
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ px: 1, pb: 1 }}>
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
          sx={{
            border: 'none',
            '& .MuiDataGrid-columnHeaders': { bgcolor: '#f8fafc' },
            '& .MuiDataGrid-row:hover': { bgcolor: '#fafbff' },
          }}
          slots={{
            noRowsOverlay: () => (
              <EmptyGridOverlay message="No experiments yet. They appear after retraining runs." />
            ),
          }}
        />
      </Paper>
    </Stack>
  )
}
