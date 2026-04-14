import { useMemo, useState } from 'react'
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

function fraudFeatureRow(): Record<string, number> {
  const o: Record<string, number> = { Amount: 100 }
  for (let i = 1; i <= 28; i++) o[`V${i}`] = 0
  return o
}

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
const SAMPLE_FRAUD = JSON.stringify([fraudFeatureRow()], null, 2)

export function InferencePage() {
  const [profile, setProfile] = useState<InferenceProfile>('adult')
  const [payloadText, setPayloadText] = useState(SAMPLE_ADULT)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [result, setResult] = useState<InferenceResponse | null>(null)

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
    setPayloadText(next === 'fraud' ? SAMPLE_FRAUD : SAMPLE_ADULT)
    setResult(null)
    setMessage(null)
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
                onClick={() => setPayloadText(profile === 'fraud' ? SAMPLE_FRAUD : SAMPLE_ADULT)}
                disabled={loading}
              >
                Load Example
              </Button>
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
