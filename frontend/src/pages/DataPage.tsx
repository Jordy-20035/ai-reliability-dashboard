import { useEffect, useState } from 'react'
import { Alert, LinearProgress, Paper, Stack, Typography } from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { getDatasets, getProvenance } from '../api/endpoints'
import type { DatasetVersion, ProvenanceRow } from '../types'
import { getErrorMessage } from '../utils/errors'

const datasetCols: GridColDef<DatasetVersion>[] = [
  { field: 'id', headerName: 'ID', width: 80 },
  { field: 'name', headerName: 'Name', minWidth: 220, flex: 1 },
  { field: 'kind', headerName: 'Kind', width: 170 },
  { field: 'row_count', headerName: 'Rows', width: 100 },
  { field: 'content_hash', headerName: 'Hash', width: 220 },
  { field: 'created_at', headerName: 'Created', width: 220 },
]

const provCols: GridColDef<ProvenanceRow>[] = [
  { field: 'id', headerName: 'ID', width: 80 },
  { field: 'lifecycle_experiment_id', headerName: 'Exp ID', width: 90 },
  { field: 'lifecycle_model_version_num', headerName: 'Model V', width: 90 },
  { field: 'dataset_version_id', headerName: 'Dataset V', width: 100 },
  { field: 'baseline_snapshot_id', headerName: 'Baseline ID', width: 100 },
  { field: 'git_sha', headerName: 'Git SHA', width: 180 },
  { field: 'created_at', headerName: 'Created', width: 220 },
]

export function DataPage() {
  const [datasets, setDatasets] = useState<DatasetVersion[]>([])
  const [provenance, setProvenance] = useState<ProvenanceRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const [d, p] = await Promise.all([getDatasets(100), getProvenance(100)])
      setDatasets(d.items)
      setProvenance(p.items)
    } catch (e) {
      setError(getErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  return (
    <Stack spacing={2}>
      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error">
          Could not load data tables — start API: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        Data Versioning & Provenance
      </Typography>
      <Paper sx={{ p: 1 }}>
        <Typography variant="h6" sx={{ p: 1 }}>
          Dataset Versions
        </Typography>
        <DataGrid
          rows={datasets}
          columns={datasetCols}
          getRowId={(r) => r.id}
          loading={loading}
          autoHeight
          pageSizeOptions={[10, 20, 50]}
          initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
        />
      </Paper>

      <Paper sx={{ p: 1 }}>
        <Typography variant="h6" sx={{ p: 1 }}>
          Training Provenance
        </Typography>
        <DataGrid
          rows={provenance}
          columns={provCols}
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

