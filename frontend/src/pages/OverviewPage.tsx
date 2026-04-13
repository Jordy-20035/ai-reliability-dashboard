import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Grid,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getOverview, runDriftCheck } from '../api/endpoints'
import { KpiCard } from '../components/KpiCard'
import type { OverviewResponse, Scenario } from '../types'
import { getErrorMessage } from '../utils/errors'

export function OverviewPage() {
  const [overview, setOverview] = useState<OverviewResponse | null>(null)
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
      const data = await getOverview()
      setOverview(data)
    } catch (e) {
      setError(getErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const chartData = useMemo(() => {
    const s = overview?.last_run?.summary
    if (!s) return []
    return [
      { name: 'High PSI', value: s.n_features_high_psi ?? 0 },
      { name: 'KS Significant', value: s.n_numeric_ks_significant ?? 0 },
      { name: 'Chi2 Significant', value: s.n_categorical_chi2_significant ?? 0 },
    ]
  }, [overview])

  async function onRunCheck() {
    if (scenario === 'incoming_csv' && !currentCsvPath.trim()) {
      setMessage('Please provide a current CSV path for incoming_csv scenario.')
      return
    }
    setMessage(null)
    setLoading(true)
    try {
      const res = await runDriftCheck(scenario, {
        currentCsvPath,
        fraudD1Path,
        fraudD2Path,
        fraudD3Path,
      })
      setMessage(
        `Drift check completed. triggered=${String(res.policy_triggered)} run_id=${String(res.run_id)}`,
      )
      await load()
    } catch (e) {
      setMessage(`Failed to run drift check: ${getErrorMessage(e)}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Stack spacing={2}>
      {loading && <LinearProgress />}
      <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 900 }}>
        Start here for a snapshot of the system. KPIs summarize how much history you have.{' '}
        <strong>Run Drift Check</strong> compares the selected scenario to the saved baseline (Adult or fraud
        D1/D2/D3 splits) and records a workflow run—use the same control on Workflows for a fuller run history.
      </Typography>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          System Overview
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
                placeholder="FRAUD_D1_PATH or path to D1.csv"
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
            aria-label="Run drift check for selected scenario"
          >
            Run Drift Check
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error">
          Could not reach API. Start backend: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}
      {message && <Alert severity="info">{message}</Alert>}

      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 3 }}>
          <KpiCard label="Workflow Runs" value={overview?.kpis.n_runs ?? 0} />
        </Grid>
        <Grid size={{ xs: 12, md: 3 }}>
          <KpiCard label="Model Versions" value={overview?.kpis.n_models ?? 0} />
        </Grid>
        <Grid size={{ xs: 12, md: 3 }}>
          <KpiCard label="Dataset Versions" value={overview?.kpis.n_datasets ?? 0} />
        </Grid>
        <Grid size={{ xs: 12, md: 3 }}>
          <KpiCard
            label="Current Production Row"
            value={overview?.kpis.production_model_row_id ?? '-'}
            subtitle="from lifecycle settings"
          />
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 8 }}>
          <Paper sx={{ p: 2, height: 320 }}>
            <Typography variant="h6" gutterBottom>
              Latest Drift Signal Counts
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={chartData}>
                <XAxis dataKey="name" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Paper sx={{ p: 2, height: 320 }}>
            <Typography variant="h6" gutterBottom>
              Last Run Summary
            </Typography>
            <Box sx={{ fontSize: 14, lineHeight: 1.8 }}>
              <div>Scenario: {overview?.last_run?.scenario ?? '-'}</div>
              <div>Triggered: {String(overview?.last_run?.policy_triggered ?? false)}</div>
              <div>Run ID: {overview?.last_run?.id ?? '-'}</div>
              <div>Latest model: v{overview?.latest_model?.version_num ?? '-'}</div>
              <div>Latest experiment: {overview?.latest_experiment?.name ?? '-'}</div>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Stack>
  )
}

