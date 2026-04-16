import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { PlayArrow } from '@mui/icons-material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { EmptyGridOverlay } from '../components/EmptyGridOverlay'
import { getRuns, runDriftCheck } from '../api/endpoints'
import type { RunRecord, Scenario } from '../types'
import { getErrorMessage } from '../utils/errors'

const cols: GridColDef<RunRecord>[] = [
  { field: 'id', headerName: 'Run ID', width: 80 },
  { field: 'started_at', headerName: 'Started', width: 190 },
  { field: 'finished_at', headerName: 'Finished', width: 190 },
  { field: 'scenario', headerName: 'Scenario', width: 160 },
  {
    field: 'policy_triggered',
    headerName: 'Triggered',
    width: 110,
    renderCell: (params) => (
      <Chip
        label={params.value ? 'Yes' : 'No'}
        size="small"
        color={params.value ? 'error' : 'success'}
        variant="outlined"
      />
    ),
  },
  {
    field: 'trigger_reasons',
    headerName: 'Trigger Reasons',
    flex: 1,
    minWidth: 260,
    valueGetter: (_, row) => row.trigger_reasons.join(' ; '),
  },
]

export function WorkflowsPage() {
  const [rows, setRows] = useState<RunRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [scenario, setScenario] = useState<Scenario>('random_holdout')
  const [currentCsvPath, setCurrentCsvPath] = useState('')
  const [fraudD1Path, setFraudD1Path] = useState('')
  const [fraudD2Path, setFraudD2Path] = useState('')
  const [fraudD3Path, setFraudD3Path] = useState('')
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const data = await getRuns(100)
      setRows(data.items)
    } catch (e) {
      setError(getErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  async function onRunCheck() {
    if (scenario === 'incoming_csv' && !currentCsvPath.trim()) {
      setMessage('Please provide a current CSV path for incoming_csv scenario.')
      return
    }
    setLoading(true)
    setMessage(null)
    try {
      const res = await runDriftCheck(scenario, {
        currentCsvPath,
        fraudD1Path,
        fraudD2Path,
        fraudD3Path,
      })
      setMessage(`Run complete (run_id=${String(res.run_id)} triggered=${String(res.policy_triggered)})`)
      await load()
    } catch (e) {
      setMessage(`Run failed: ${getErrorMessage(e)}`)
      setLoading(false)
    }
  }

  return (
    <Stack spacing={2.5}>
      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error">
          Could not load runs — start API: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}
      {message && <Alert severity="info">{message}</Alert>}

      {/* ---- Compact quick-run bar ---- */}
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
        <Typography variant="body2" color="text.secondary" sx={{ mr: 0.5 }}>
          Quick run:
        </Typography>
        <Select size="small" value={scenario} onChange={(e) => setScenario(e.target.value as Scenario)} sx={{ minWidth: 160 }}>
          <MenuItem value="random_holdout">random_holdout</MenuItem>
          <MenuItem value="age_shift">age_shift</MenuItem>
          <MenuItem value="incoming_csv">incoming_csv</MenuItem>
          <MenuItem value="fraud_d1_vs_d2">fraud_d1_vs_d2</MenuItem>
          <MenuItem value="fraud_d2_vs_d3">fraud_d2_vs_d3</MenuItem>
          <MenuItem value="fraud_d1_vs_d3">fraud_d1_vs_d3</MenuItem>
        </Select>
        {scenario === 'incoming_csv' && (
          <TextField size="small" label="CSV path" value={currentCsvPath} onChange={(e) => setCurrentCsvPath(e.target.value)} sx={{ minWidth: 220 }} />
        )}
        {scenario.startsWith('fraud_') && (
          <>
            <TextField size="small" label="D1" value={fraudD1Path} onChange={(e) => setFraudD1Path(e.target.value)} sx={{ width: 130 }} />
            <TextField size="small" label="D2" value={fraudD2Path} onChange={(e) => setFraudD2Path(e.target.value)} sx={{ width: 130 }} />
            <TextField size="small" label="D3" value={fraudD3Path} onChange={(e) => setFraudD3Path(e.target.value)} sx={{ width: 130 }} />
          </>
        )}
        <Button variant="contained" size="small" startIcon={<PlayArrow />} onClick={() => void onRunCheck()} disabled={loading}>
          Run
        </Button>
      </Box>

      {/* ---- Runs table ---- */}
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ px: 1, pb: 1 }}>
          Orchestration Runs
        </Typography>
        <DataGrid
          rows={rows}
          columns={cols}
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
              <EmptyGridOverlay message="No workflow runs yet. Run a drift check from Overview or the shortcut above." />
            ),
          }}
        />
      </Paper>
    </Stack>
  )
}
