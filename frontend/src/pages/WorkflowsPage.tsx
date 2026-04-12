import { useEffect, useState } from 'react'
import { Alert, Box, Button, LinearProgress, MenuItem, Paper, Select, Stack, TextField, Typography } from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { EmptyGridOverlay } from '../components/EmptyGridOverlay'
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
    <Stack spacing={2}>
      {loading && <LinearProgress />}
      <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 900 }}>
        Each row is one <strong>orchestration run</strong>: drift (and related checks) plus whether the{' '}
        <strong>policy triggered</strong> (the automation hook that can lead to retraining). You can run a new
        check here as an operator shortcut; it is the <strong>same check</strong> used on Overview.
      </Typography>
      {error && (
        <Alert severity="error">
          Could not load runs — start API: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Workflow Tracking
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <Select
            size="small"
            value={scenario}
            onChange={(e) => setScenario(e.target.value as Scenario)}
          >
            <MenuItem value="random_holdout">random_holdout</MenuItem>
            <MenuItem value="age_shift">age_shift</MenuItem>
            <MenuItem value="incoming_csv">incoming_csv</MenuItem>
            <MenuItem value="fraud_d1_vs_d2">fraud_d1_vs_d2</MenuItem>
            <MenuItem value="fraud_d2_vs_d3">fraud_d2_vs_d3</MenuItem>
            <MenuItem value="fraud_d1_vs_d3">fraud_d1_vs_d3</MenuItem>
          </Select>
          {scenario === 'incoming_csv' && (
            <TextField
              size="small"
              label="Current CSV Path"
              placeholder="./data/raw/adult.csv"
              value={currentCsvPath}
              onChange={(e) => setCurrentCsvPath(e.target.value)}
              sx={{ minWidth: 300 }}
            />
          )}
          {scenario.startsWith('fraud_') && (
            <>
              <TextField
                size="small"
                label="D1 CSV (optional)"
                placeholder="FRAUD_D1_PATH"
                value={fraudD1Path}
                onChange={(e) => setFraudD1Path(e.target.value)}
                sx={{ minWidth: 220 }}
              />
              <TextField
                size="small"
                label="D2 CSV (optional)"
                placeholder="FRAUD_D2_PATH"
                value={fraudD2Path}
                onChange={(e) => setFraudD2Path(e.target.value)}
                sx={{ minWidth: 220 }}
              />
              <TextField
                size="small"
                label="D3 CSV (optional)"
                placeholder="FRAUD_D3_PATH"
                value={fraudD3Path}
                onChange={(e) => setFraudD3Path(e.target.value)}
                sx={{ minWidth: 220 }}
              />
            </>
          )}
          <Button
            variant="contained"
            onClick={() => void onRunCheck()}
            disabled={loading}
            aria-label="Run new workflow check"
          >
            Run New Workflow
          </Button>
        </Box>
      </Box>
      <Typography variant="caption" color="text.secondary">
        Same action as Overview → Run Drift Check; results are written to this table.
      </Typography>
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
          slots={{
            noRowsOverlay: () => (
              <EmptyGridOverlay message="No workflow runs yet. Use Trigger Check (or Overview → Run Drift Check) after the API is running." />
            ),
          }}
        />
      </Paper>
    </Stack>
  )
}

