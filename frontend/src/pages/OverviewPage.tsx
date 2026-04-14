import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  Grid,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import {
  Analytics,
  ModelTraining,
  PlayArrow,
  Settings,
  TrendingUp,
} from '@mui/icons-material'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
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
      { name: 'Chi² Significant', value: s.n_categorical_chi2_significant ?? 0 },
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

  const triggered = overview?.last_run?.policy_triggered

  return (
    <Stack spacing={2.5}>
      {loading && <LinearProgress />}

      {error && (
        <Alert severity="error">
          Could not reach API. Start backend: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}
      {message && <Alert severity="info">{message}</Alert>}

      {/* ---- KPI row ---- */}
      <Grid container spacing={2}>
        <Grid size={{ xs: 6, md: 3 }}>
          <KpiCard
            icon={<TrendingUp fontSize="small" />}
            label="Drift score (mean PSI)"
            value={
              overview?.last_run?.summary?.mean_psi != null
                ? (overview.last_run.summary.mean_psi as number).toFixed(3)
                : '-'
            }
            subtitle={triggered != null ? (triggered ? 'policy triggered' : 'within threshold') : undefined}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <KpiCard
            icon={<Analytics fontSize="small" />}
            label="Workflow Runs"
            value={overview?.kpis.n_runs ?? 0}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <KpiCard
            icon={<ModelTraining fontSize="small" />}
            label="Model Versions"
            value={overview?.kpis.n_models ?? 0}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <KpiCard
            icon={<Settings fontSize="small" />}
            label="Production Row"
            value={overview?.kpis.production_model_row_id ?? '-'}
            subtitle="serving pointer"
          />
        </Grid>
      </Grid>

      {/* ---- Chart + summary ---- */}
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 8 }}>
          <Paper variant="outlined" sx={{ p: 2.5, height: 340 }}>
            <Typography variant="h6" gutterBottom>
              Drift Signal Counts
            </Typography>
            <ResponsiveContainer width="100%" height="88%">
              <BarChart data={chartData} barCategoryGap="25%">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#1a5fb4" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Paper variant="outlined" sx={{ p: 2.5, height: 340 }}>
            <Typography variant="h6" gutterBottom>
              Last Run
            </Typography>
            <Stack spacing={1.25} sx={{ fontSize: 14 }}>
              <Row label="Scenario" val={overview?.last_run?.scenario} />
              <Row
                label="Triggered"
                val={
                  triggered != null ? (
                    <Chip
                      label={triggered ? 'Yes' : 'No'}
                      size="small"
                      color={triggered ? 'error' : 'success'}
                      variant="outlined"
                    />
                  ) : (
                    '-'
                  )
                }
              />
              <Row label="Run ID" val={overview?.last_run?.id} />
              <Row label="Latest model" val={overview?.latest_model ? `v${overview.latest_model.version_num}` : undefined} />
              <Row label="Experiment" val={overview?.latest_experiment?.name} />
            </Stack>
          </Paper>
        </Grid>
      </Grid>

      {/* ---- Action buttons ---- */}
      <Paper variant="outlined" sx={{ p: 2.5 }}>
        <Typography variant="h6" gutterBottom>
          Run Drift Check
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
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
              sx={{ minWidth: 260 }}
            />
          )}
          {scenario.startsWith('fraud_') && (
            <>
              <TextField size="small" label="D1 (opt)" value={fraudD1Path} onChange={(e) => setFraudD1Path(e.target.value)} sx={{ minWidth: 180 }} />
              <TextField size="small" label="D2 (opt)" value={fraudD2Path} onChange={(e) => setFraudD2Path(e.target.value)} sx={{ minWidth: 180 }} />
              <TextField size="small" label="D3 (opt)" value={fraudD3Path} onChange={(e) => setFraudD3Path(e.target.value)} sx={{ minWidth: 180 }} />
            </>
          )}
          <Button
            variant="contained"
            startIcon={<PlayArrow />}
            onClick={() => void onRunCheck()}
            disabled={loading}
          >
            Run Check
          </Button>
        </Box>
      </Paper>
    </Stack>
  )
}

function Row({ label, val }: { label: string; val?: React.ReactNode }) {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" sx={{ fontWeight: 600 }}>
        {val ?? '-'}
      </Typography>
    </Box>
  )
}
