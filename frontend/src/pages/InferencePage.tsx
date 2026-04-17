import { useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  Grid,
  LinearProgress,
  Menu,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { PlayArrow, RestartAlt } from '@mui/icons-material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { predictProduction } from '../api/endpoints'
import type { InferenceResponse } from '../types'
import { getErrorMessage } from '../utils/errors'

type InferenceProfile = 'adult' | 'fraud'

type PredictionRow = {
  id: number
  prediction: number | string
  class_label: string
  positive_probability: string
}

const SAMPLE_FRAUD_TEMPLATE_1 = JSON.stringify(
  [
    {
      V1: -0.85,
      V2: 0.72,
      V3: -0.68,
      V4: 1.25,
      V5: -0.18,
      V6: -0.35,
      V7: -0.92,
      V8: 0.41,
      V9: -0.88,
      V10: -0.74,
      V11: 1.05,
      V12: -0.95,
      V13: -0.12,
      V14: -1.1,
      V15: 0.18,
      V16: -0.32,
      V17: -0.85,
      V18: -0.08,
      V19: 0.16,
      V20: 0.05,
      V21: 0.12,
      V22: -0.02,
      V23: -0.1,
      V24: 0.08,
      V25: 0.03,
      V26: 0.06,
      V27: 0.09,
      V28: 0.04,
      Amount: 89.5,
    },
  ],
  null,
  2,
)

const SAMPLE_FRAUD_TEMPLATE_2 = JSON.stringify(
  [
    {
      V1: -1.65,
      V2: 1.35,
      V3: -1.05,
      V4: 2.25,
      V5: -0.35,
      V6: -0.95,
      V7: -1.55,
      V8: 0.85,
      V9: -1.6,
      V10: -1.55,
      V11: 2.15,
      V12: -1.85,
      V13: -0.4,
      V14: -2.6,
      V15: 0.28,
      V16: -0.75,
      V17: -1.75,
      V18: -0.05,
      V19: 0.3,
      V20: 0.09,
      V21: 0.35,
      V22: -0.03,
      V23: -0.28,
      V24: 0.22,
      V25: -0.03,
      V26: 0.16,
      V27: 0.24,
      V28: 0.08,
      Amount: 149.62,
    },
  ],
  null,
  2,
)

const SAMPLE_ADULT = JSON.stringify(
  [
    {
      age: 37, fnlwgt: 120000, 'education.num': 10, 'capital.gain': 0,
      'capital.loss': 0, 'hours.per.week': 40, workclass: 'Private',
      education: 'HS-grad', 'marital.status': 'Married-civ-spouse',
      occupation: 'Craft-repair', relationship: 'Husband', race: 'White',
      sex: 'Male', 'native.country': 'United-States',
    },
  ],
  null,
  2,
)

export function InferencePage() {
  const [profile, setProfile] = useState<InferenceProfile>('adult')
  const [payloadText, setPayloadText] = useState(SAMPLE_ADULT)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [result, setResult] = useState<InferenceResponse | null>(null)
  const [fraudExampleAnchor, setFraudExampleAnchor] = useState<null | HTMLElement>(null)

  const cols = useMemo((): GridColDef<PredictionRow>[] => {
    const probH = profile === 'fraud' ? 'P(fraud)' : 'P(>50K)'
    const classH = profile === 'fraud' ? 'Label' : 'Income Class'
    return [
      { field: 'id', headerName: 'Row', width: 70 },
      { field: 'prediction', headerName: 'Pred', width: 90 },
      { field: 'class_label', headerName: classH, width: 180 },
      { field: 'positive_probability', headerName: probH, width: 110 },
    ]
  }, [profile])

  const rows = useMemo<PredictionRow[]>(() => {
    if (!result) return []
    return result.predictions.map((pred, idx) => {
      const label =
        result.profile === 'fraud'
          ? (result.predicted_fraud_label?.[idx] ?? '-')
          : (result.predicted_income_class?.[idx] ?? '-')
      return {
        id: idx + 1,
        prediction: pred,
        class_label: label,
        positive_probability:
          result.positive_class_probability?.[idx] != null
            ? result.positive_class_probability[idx].toFixed(4)
            : '-',
      }
    })
  }, [result])

  function onProfileChange(next: InferenceProfile) {
    setProfile(next)
    setPayloadText(next === 'fraud' ? SAMPLE_FRAUD_TEMPLATE_1 : SAMPLE_ADULT)
    setResult(null)
    setMessage(null)
  }

  function loadExampleAdult() {
    setPayloadText(SAMPLE_ADULT)
  }

  function openFraudExamplesMenu(el: HTMLElement) {
    setFraudExampleAnchor(el)
  }

  function closeFraudExamplesMenu() {
    setFraudExampleAnchor(null)
  }

  function loadExampleFraud(template: 1 | 2) {
    setPayloadText(template === 1 ? SAMPLE_FRAUD_TEMPLATE_1 : SAMPLE_FRAUD_TEMPLATE_2)
    closeFraudExamplesMenu()
  }

  async function onPredict() {
    setMessage(null)
    setLoading(true)
    setResult(null)
    try {
      const parsed = JSON.parse(payloadText)
      if (!Array.isArray(parsed)) {
        setMessage('Payload must be a JSON array of row objects.')
        setLoading(false)
        return
      }
      const res = await predictProduction(parsed, profile)
      setResult(res)
    } catch (e) {
      setMessage(getErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  const topProb =
    result?.positive_class_probability?.[0] != null
      ? result.positive_class_probability[0]
      : null
  const scorePercent = topProb != null ? Math.round(topProb * 100) : null

  return (
    <Stack spacing={2.5}>
      {loading && <LinearProgress />}
      {message && <Alert severity="error">{message}</Alert>}

      {/* ---- SPLIT PANE ---- */}
      <Grid container spacing={2}>
        {/* LEFT: input */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper variant="outlined" sx={{ p: 2.5, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
              <Typography variant="h6">Input (JSON)</Typography>
              <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                <Typography variant="caption" color="text.secondary">Profile</Typography>
                <Select
                  size="small"
                  value={profile}
                  onChange={(e) => onProfileChange(e.target.value as InferenceProfile)}
                  sx={{ minWidth: 100 }}
                >
                  <MenuItem value="adult">adult</MenuItem>
                  <MenuItem value="fraud">fraud</MenuItem>
                </Select>
              </Stack>
            </Box>
            <TextField
              multiline
              minRows={14}
              maxRows={22}
              value={payloadText}
              onChange={(e) => setPayloadText(e.target.value)}
              fullWidth
              sx={{
                flexGrow: 1,
                '& .MuiInputBase-root': {
                  fontFamily: 'monospace',
                  fontSize: '0.82rem',
                  bgcolor: '#f8fafc',
                },
              }}
            />
            <Box sx={{ display: 'flex', gap: 1.5, mt: 2 }}>
              <Button variant="contained" startIcon={<PlayArrow />} onClick={() => void onPredict()} disabled={loading}>
                Submit Inference
              </Button>
              <Button
                variant="outlined"
                startIcon={<RestartAlt />}
                onClick={(e) =>
                  profile === 'fraud'
                    ? openFraudExamplesMenu(e.currentTarget)
                    : loadExampleAdult()
                }
                disabled={loading}
              >
                Load Example
              </Button>
              <Menu
                anchorEl={fraudExampleAnchor}
                open={Boolean(fraudExampleAnchor)}
                onClose={closeFraudExamplesMenu}
                disableAutoFocusItem
              >
                <MenuItem onClick={() => loadExampleFraud(1)}>Fraud template 1 (softer)</MenuItem>
                <MenuItem onClick={() => loadExampleFraud(2)}>Fraud template 2 (strong)</MenuItem>
              </Menu>
            </Box>
          </Paper>
        </Grid>

        {/* RIGHT: results */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper variant="outlined" sx={{ p: 2.5, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Results
            </Typography>

            {result ? (
              <Stack spacing={2} sx={{ flexGrow: 1 }}>
                {/* Score gauge-like block */}
                {scorePercent != null && (
                  <Box sx={{ textAlign: 'center', py: 1.5 }}>
                    <Typography variant="h2" sx={{ fontWeight: 800, color: 'primary.main' }}>
                      {scorePercent}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                      Confidence (P positive class)
                    </Typography>
                    <Box
                      sx={{
                        mt: 1.5,
                        mx: 'auto',
                        width: '80%',
                        height: 10,
                        borderRadius: 5,
                        bgcolor: '#e2e8f0',
                        overflow: 'hidden',
                      }}
                    >
                      <Box
                        sx={{
                          height: '100%',
                          width: `${scorePercent}%`,
                          borderRadius: 5,
                          bgcolor: scorePercent > 50 ? '#1a5fb4' : '#16a34a',
                          transition: 'width 0.4s',
                        }}
                      />
                    </Box>
                  </Box>
                )}

                <Divider />

                {/* Model info */}
                <Box sx={{ display: 'grid', gridTemplateColumns: '110px 1fr', gap: 0.5, fontSize: 13 }}>
                  <Typography variant="caption" color="text.secondary">Model</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>Row {result.model_row_id} · v{result.model_version_num}</Typography>
                  <Typography variant="caption" color="text.secondary">Profile</Typography>
                  <Chip label={result.profile} size="small" variant="outlined" color={result.profile === 'fraud' ? 'error' : 'primary'} />
                  <Typography variant="caption" color="text.secondary">Rows scored</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{result.n_rows}</Typography>
                </Box>

                <Divider />

                {/* Results table */}
                <DataGrid
                  rows={rows}
                  columns={cols}
                  autoHeight
                  pageSizeOptions={[10, 20]}
                  initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
                  sx={{
                    border: 'none',
                    '& .MuiDataGrid-columnHeaders': { bgcolor: '#f8fafc' },
                  }}
                />
              </Stack>
            ) : (
              <Box
                sx={{
                  flexGrow: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'text.secondary',
                }}
              >
                <Typography variant="body2">Submit inference to see results here.</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Stack>
  )
}
