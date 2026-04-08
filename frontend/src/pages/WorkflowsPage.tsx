import { useEffect, useState } from 'react'
import { Alert, Box, Button, LinearProgress, MenuItem, Paper, Select, Stack, Typography } from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { getRuns, runDriftCheck } from '../api/endpoints'
import type { RunRecord, Scenario } from '../types'
import { getErrorMessage } from '../utils/errors'

const cols: GridColDef<RunRecord>[] = [
  { field: 'id', headerName: 'Run ID', width: 90 },
  { field: 'started_at', headerName: 'Started', width: 220 },
  { field: 'finished_at', headerName: 'Finished', width: 220 },
  { field: 'scenario', headerName: 'Scenario', width: 160 },
  {
    field: 'policy_triggered',
    headerName: 'Triggered',
    width: 110,
    valueFormatter: (v) => (v ? 'Yes' : 'No'),
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
    setLoading(true)
    setMessage(null)
    try {
      const res = await runDriftCheck(scenario)
      setMessage(`Run complete (run_id=${String(res.run_id)} triggered=${String(res.policy_triggered)})`)
      await load()
    } catch (e) {
      setMessage(`Run failed: ${getErrorMessage(e)}`)
      setLoading(false)
    }
  }

  return (
    <Stack spacing={2}>
      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error">
          Could not load runs — start API: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Workflow Tracking
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Select
            size="small"
            value={scenario}
            onChange={(e) => setScenario(e.target.value as Scenario)}
          >
            <MenuItem value="random_holdout">random_holdout</MenuItem>
            <MenuItem value="age_shift">age_shift</MenuItem>
          </Select>
          <Button variant="contained" onClick={() => void onRunCheck()} disabled={loading}>
            Trigger Check
          </Button>
        </Box>
      </Box>
      {message && <Alert severity="info">{message}</Alert>}

      <Paper sx={{ p: 1 }}>
        <DataGrid
          rows={rows}
          columns={cols}
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

