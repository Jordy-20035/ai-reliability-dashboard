import { useEffect, useState } from 'react'
import { Alert, LinearProgress, Paper, Stack, Typography } from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { EmptyGridOverlay } from '../components/EmptyGridOverlay'
import { getDatasets, getProvenance } from '../api/endpoints'
import type { DatasetVersion, ProvenanceRow } from '../types'
import { getErrorMessage } from '../utils/errors'

const datasetCols: GridColDef<DatasetVersion>[] = [
  { field: 'id', headerName: 'ID', width: 70 },
  { field: 'name', headerName: 'Name', minWidth: 220, flex: 1 },
  { field: 'kind', headerName: 'Kind', width: 160 },
  { field: 'row_count', headerName: 'Rows', width: 90 },
  { field: 'content_hash', headerName: 'Hash', width: 200 },
  { field: 'created_at', headerName: 'Created', width: 190 },
]

const provCols: GridColDef<ProvenanceRow>[] = [
  { field: 'id', headerName: 'ID', width: 70 },
  { field: 'lifecycle_experiment_id', headerName: 'Exp ID', width: 80 },
  { field: 'lifecycle_model_version_num', headerName: 'Model V', width: 90 },
  { field: 'dataset_version_id', headerName: 'Dataset V', width: 90 },
  { field: 'baseline_snapshot_id', headerName: 'Baseline', width: 90 },
  { field: 'git_sha', headerName: 'Git SHA', width: 160 },
  { field: 'created_at', headerName: 'Created', width: 190 },
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
    <Stack spacing={2.5}>
      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error">
          Could not load data — start API: <code>python -m src.api --port 8000</code> — {error}
        </Alert>
      )}

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ px: 1, pb: 1 }}>
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
          sx={{
            border: 'none',
            '& .MuiDataGrid-columnHeaders': { bgcolor: '#f8fafc' },
            '& .MuiDataGrid-row:hover': { bgcolor: '#fafbff' },
          }}
          slots={{
            noRowsOverlay: () => (
              <EmptyGridOverlay message="No dataset versions recorded yet." />
            ),
          }}
        />
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ px: 1, pb: 1 }}>
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
          sx={{
            border: 'none',
            '& .MuiDataGrid-columnHeaders': { bgcolor: '#f8fafc' },
            '& .MuiDataGrid-row:hover': { bgcolor: '#fafbff' },
          }}
          slots={{
            noRowsOverlay: () => (
              <EmptyGridOverlay message="No provenance rows yet." />
            ),
          }}
        />
      </Paper>
    </Stack>
  )
}
