import { Close, HelpOutlined } from '@mui/icons-material'
import {
  Box,
  Button,
  Divider,
  Drawer,
  IconButton,
  Link,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Toolbar,
  Typography,
} from '@mui/material'
import { useState, type ReactNode } from 'react'
import { Link as RouterLink } from 'react-router-dom'

type HelpDrawerProps = {
  /** Anchor element for drawer width on large screens */
  wide?: boolean
  /** Render trigger as floating fixed button (bottom-right). */
  floating?: boolean
}

export function HelpDrawer({ wide = true, floating = false }: HelpDrawerProps) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <Box
        sx={
          floating
            ? {
                position: 'fixed',
                right: 20,
                bottom: 20,
                zIndex: (theme) => theme.zIndex.drawer - 1,
              }
            : undefined
        }
      >
        <Button
          color={floating ? 'inherit' : 'inherit'}
          variant={floating ? 'outlined' : 'text'}
          startIcon={<HelpOutlined />}
          onClick={() => setOpen(true)}
          size="small"
          sx={
            floating
              ? {
                  borderRadius: 999,
                  bgcolor: 'background.paper',
                  borderColor: 'divider',
                  boxShadow: 2,
                  px: 1.75,
                }
              : { border: '1px solid rgba(255,255,255,0.35)' }
          }
        >
          How it works
        </Button>
      </Box>
      <Drawer anchor="right" open={open} onClose={() => setOpen(false)}>
        <Toolbar
          sx={{
            justifyContent: 'space-between',
            borderBottom: '1px solid',
            borderColor: 'divider',
            minHeight: 56,
          }}
        >
          <Typography variant="h6" component="span" sx={{ fontWeight: 700 }}>
            How this platform fits together
          </Typography>
          <IconButton aria-label="Close help" onClick={() => setOpen(false)} edge="end">
            <Close />
          </IconButton>
        </Toolbar>
        <Box
          sx={{
            width: wide ? { xs: '100vw', sm: 440, md: 520 } : 360,
            maxWidth: '100vw',
            p: 2,
            pb: 4,
            overflow: 'auto',
          }}
        >
          <Stack spacing={2.5}>
            <Section title="Dataset or model first?">
              <Typography variant="body2" color="text.secondary">
                In real systems you have <strong>both</strong>: data you train and score on, and the model
                artifact plus metrics. You do not pick one. This app wires a common story:{' '}
                <strong>baseline data → drift signal → orchestration → retrain → registered version →
                promotion</strong>. The <RouterLink to="/data">Data</RouterLink> page shows{' '}
                <em>which</em> snapshots and hashes were used; the <RouterLink to="/models">Models</RouterLink>{' '}
                page shows <em>which</em> version is candidate vs production.
              </Typography>
            </Section>

            <Section title="A full pass through the UI (demo order)">
              <Typography variant="body2" color="text.secondary" component="div">
                <Box component="ol" sx={{ pl: 2.25, m: 0, '& li': { mb: 1 } }}>
                  <li>
                    <RouterLink to="/">Overview</RouterLink> — KPIs and latest drift counts. Use{' '}
                    <strong>Run Drift Check</strong> to compare a live slice to the baseline (pick a
                    scenario). Same action exists on Workflows.
                  </li>
                  <li>
                    <RouterLink to="/workflows">Workflows</RouterLink> — Every orchestration run: time, scenario,
                    whether policy <strong>triggered</strong> (the hook that can lead to retrain).
                  </li>
                  <li>
                    <RouterLink to="/models">Models</RouterLink> — <strong>Trigger Retraining</strong> runs the
                    training pipeline and registers a new row. Copy a <strong>Row ID</strong> from the table,
                    then <strong>Promote stage</strong> to move staging → production (or archive).
                  </li>
                  <li>
                    <RouterLink to="/data">Data</RouterLink> — Read-only: dataset versions (hashes) and{' '}
                    <strong>provenance</strong> linking experiments, model versions, and data snapshots.
                  </li>
                </Box>
              </Typography>
            </Section>

            <Section title="Flow (one glance)">
              <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'action.hover' }}>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 13, lineHeight: 1.7 }}>
                  Data baseline &amp; snapshots → Drift checks (Overview / Workflows) → Policy &amp; runs →
                  Retrain (Models) → Lifecycle &amp; promote → Production pointer
                </Typography>
              </Paper>
            </Section>

            <Section title="How this relates to common MLOps tools">
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                This project is a <strong>unified thesis-style stack</strong>, not a feature-complete clone of
                every vendor. Use the table as a mental map, not a sales claim.
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Typical product</TableCell>
                      <TableCell>Closest piece here</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Evidently AI, NannyML, Alibi Detect</TableCell>
                      <TableCell>Drift metrics &amp; checks; Overview / Workflows runs</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>MLflow (Registry)</TableCell>
                      <TableCell>Model versions, stages, promote; production pointer</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Weights &amp; Biases</TableCell>
                      <TableCell>Experiments grid &amp; metrics columns (lighter than full W&amp;B)</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Fiddler AI</TableCell>
                      <TableCell>Monitoring slice only; no full fairness / XAI / business console</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Section>

            <Section title="Glossary (plain English)">
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Term</TableCell>
                      <TableCell>What it means here</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Scenario</TableCell>
                      <TableCell>How current/incoming data is chosen for a check or retrain run.</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>random_holdout</TableCell>
                      <TableCell>Reference/current are random splits from same dataset (mostly healthy case).</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>age_shift</TableCell>
                      <TableCell>Current data is intentionally shifted (age &gt;= 40) to simulate drift.</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>incoming_csv</TableCell>
                      <TableCell>Current data comes from an external CSV batch path (real incoming mode).</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Run Drift Check / Run New Workflow</TableCell>
                      <TableCell>Same backend action from two pages; each click creates one run record.</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Trigger Retraining</TableCell>
                      <TableCell>Manual retrain action (separate from auto retrain when policy triggers).</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>development stage</TableCell>
                      <TableCell>Newly created candidate model version; not approved for live traffic yet.</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>staging stage</TableCell>
                      <TableCell>Pre-production validation stage before final promotion.</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>production stage</TableCell>
                      <TableCell>Version marked as the live model by lifecycle production pointer.</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>archived stage</TableCell>
                      <TableCell>Retired version kept for history and audit.</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Production pointer</TableCell>
                      <TableCell>The model row ID the system treats as current production.</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                Note: Models retraining currently supports random_holdout and age_shift only; incoming_csv is
                available for drift/orchestration checks.
              </Typography>
            </Section>

            <Divider />

            <Typography variant="body2" color="text.secondary">
              API reference:{' '}
              <Link href="/docs" target="_blank" rel="noreferrer">
                OpenAPI docs
              </Link>
              . Backend example: <code>python -m src.api --port 8000</code>
            </Typography>
          </Stack>
        </Box>
      </Drawer>
    </>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Box>
      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 0.75 }}>
        {title}
      </Typography>
      {children}
    </Box>
  )
}
