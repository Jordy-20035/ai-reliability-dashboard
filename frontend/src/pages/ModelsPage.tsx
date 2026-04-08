import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { getExperiments, getModels, getProductionPointer, promoteModel, runRetrain } from '../api/endpoints'
import type { LifecycleExperiment, LifecycleModel, Scenario } from '../types'

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
  const [scenario, setScenario] = useState<Scenario>('random_holdout')
  const [promoteId, setPromoteId] = useState('')
  const [promoteStage, setPromoteStage] = useState('staging')
  const [message, setMessage] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    try {
      const [m, e, p] = await Promise.all([getModels(), getExperiments(100), getProductionPointer()])
      setModels(m.items)
      setExperiments(e.items)
      setProductionId(p.production_model_row_id)
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
      setMessage(`Retrain failed: ${String(e)}`)
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
      setMessage(`Promotion failed: ${String(e)}`)
      setLoading(false)
    }
  }

  return (
    <Stack spacing={2}>
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        Model Lifecycle & Control
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Current production model row id: {productionId ?? '-'}
      </Typography>
      {message && <Alert severity="info">{message}</Alert>}

      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
        <Select
          size="small"
          value={scenario}
          onChange={(e) => setScenario(e.target.value as Scenario)}
        >
          <MenuItem value="random_holdout">random_holdout</MenuItem>
          <MenuItem value="age_shift">age_shift</MenuItem>
        </Select>
        <Button variant="contained" onClick={() => void onRetrain()} disabled={loading}>
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
        <Button variant="outlined" onClick={() => void onPromote()} disabled={loading}>
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
        />
      </Paper>
    </Stack>
  )
}

