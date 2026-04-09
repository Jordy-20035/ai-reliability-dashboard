import { useMemo, useState } from 'react'
import { Alert, Box, Button, LinearProgress, Paper, Stack, TextField, Typography } from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { predictProduction } from '../api/endpoints'
import type { InferenceResponse } from '../types'
import { getErrorMessage } from '../utils/errors'

type PredictionRow = {
  id: number
  prediction: number | string
  income_class: string
  positive_probability: string
}

const SAMPLE_INPUT = JSON.stringify(
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

const cols: GridColDef<PredictionRow>[] = [
  { field: 'id', headerName: 'Row', width: 90 },
  { field: 'prediction', headerName: 'Prediction', width: 130 },
  { field: 'income_class', headerName: 'Predicted Income Class', width: 220 },
  { field: 'positive_probability', headerName: 'P(>50K)', width: 130 },
]

export function InferencePage() {
  const [payloadText, setPayloadText] = useState(SAMPLE_INPUT)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [result, setResult] = useState<InferenceResponse | null>(null)

  const rows = useMemo<PredictionRow[]>(() => {
    if (!result) return []
    return result.predictions.map((pred, idx) => ({
      id: idx + 1,
      prediction: pred,
      income_class: result.predicted_income_class[idx] ?? '-',
      positive_probability:
        result.positive_class_probability?.[idx] != null
          ? result.positive_class_probability[idx].toFixed(4)
          : '-',
    }))
  }, [result])

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
      const res = await predictProduction(parsed)
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
        Score data against the current production model pointer. Paste JSON rows with Adult feature
        columns and run inference.
      </Typography>
      <Typography variant="caption" color="text.secondary">
        Tip: Use API docs for schema details if you customize the payload.
      </Typography>
      {message && <Alert severity="error">{message}</Alert>}

      <Paper sx={{ p: 2 }}>
        <Stack spacing={1.5}>
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
            <Button variant="outlined" onClick={() => setPayloadText(SAMPLE_INPUT)} disabled={loading}>
              Reset Sample
            </Button>
          </Box>
        </Stack>
      </Paper>

      {result && (
        <Paper sx={{ p: 1 }}>
          <Typography variant="body2" sx={{ p: 1 }}>
            Production model row <strong>{result.model_row_id}</strong> (version{' '}
            <strong>{result.model_version_num}</strong>) scored <strong>{result.n_rows}</strong> row(s).
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
