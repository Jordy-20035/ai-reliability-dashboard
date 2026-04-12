import { useMemo, useState } from 'react'
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
  for (let i = 1; i <= 28; i++) {
    o[`V${i}`] = 0
  }
  return o
}

const SAMPLE_ADULT = JSON.stringify(
  [
    {
      age: 37,
      fnlwgt: 120000,
      'education.num': 10,
      'capital.gain': 0,
      'capital.loss': 0,
      'hours.per.week': 40,
      workclass: 'Private',
      education: 'HS-grad',
      'marital.status': 'Married-civ-spouse',
      occupation: 'Craft-repair',
      relationship: 'Husband',
      race: 'White',
      sex: 'Male',
      'native.country': 'United-States',
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
    const probHeader = profile === 'fraud' ? 'P(fraud)' : 'P(>50K)'
    const classHeader = profile === 'fraud' ? 'Fraud label' : 'Predicted income class'
    return [
      { field: 'id', headerName: 'Row', width: 90 },
      { field: 'prediction', headerName: 'Prediction', width: 130 },
      { field: 'class_label', headerName: classHeader, width: 220 },
      { field: 'positive_probability', headerName: probHeader, width: 130 },
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

  return (
    <Stack spacing={2}>
      {loading && <LinearProgress />}
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        Production Inference
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 980 }}>
        Score rows against the production model pointer. Choose <strong>Adult</strong> (census features) or{' '}
        <strong>Fraud</strong> (V1–V28 + Amount) so the API validates the correct columns for your trained
        artifact.
      </Typography>
      <Typography variant="caption" color="text.secondary">
        Tip: Use API docs for schema details if you customize the payload.
      </Typography>
      {message && <Alert severity="error">{message}</Alert>}

      <Paper sx={{ p: 2 }}>
        <Stack spacing={1.5}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Typography variant="body2" color="text.secondary">
              Feature profile
            </Typography>
            <Select
              size="small"
              value={profile}
              onChange={(e) => onProfileChange(e.target.value as InferenceProfile)}
            >
              <MenuItem value="adult">adult</MenuItem>
              <MenuItem value="fraud">fraud</MenuItem>
            </Select>
          </Box>
          <TextField
            label="Rows JSON payload"
            multiline
            minRows={10}
            value={payloadText}
            onChange={(e) => setPayloadText(e.target.value)}
            fullWidth
          />
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Button variant="contained" onClick={() => void onPredict()} disabled={loading}>
              Run Production Inference
            </Button>
            <Button
              variant="outlined"
              onClick={() => setPayloadText(profile === 'fraud' ? SAMPLE_FRAUD : SAMPLE_ADULT)}
              disabled={loading}
            >
              Reset Sample
            </Button>
          </Box>
        </Stack>
      </Paper>

      {result && (
        <Paper sx={{ p: 1 }}>
          <Typography variant="body2" sx={{ p: 1 }}>
            Production model row <strong>{result.model_row_id}</strong> (version{' '}
            <strong>{result.model_version_num}</strong>) scored <strong>{result.n_rows}</strong> row(s). Profile:{' '}
            <strong>{result.profile}</strong>.
          </Typography>
          <DataGrid
            rows={rows}
            columns={cols}
            autoHeight
            pageSizeOptions={[10, 20, 50]}
            initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
          />
        </Paper>
      )}
    </Stack>
  )
}
