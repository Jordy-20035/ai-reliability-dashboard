import { Analytics, Dataset, Hub, ModelTraining } from '@mui/icons-material'
import {
  AppBar,
  Box,
  Button,
  Container,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { HelpDrawer } from './HelpDrawer'

const nav = [
  {
    label: 'Overview',
    to: '/',
    icon: <Analytics fontSize="small" />,
    hint: 'KPIs, latest drift counts, run a drift check against the baseline',
  },
  {
    label: 'Workflows',
    to: '/workflows',
    icon: <Hub fontSize="small" />,
    hint: 'History of orchestration runs and whether policy triggered',
  },
  {
    label: 'Models',
    to: '/models',
    icon: <ModelTraining fontSize="small" />,
    hint: 'Retrain, promote stages, see experiments and production pointer',
  },
  {
    label: 'Data',
    to: '/data',
    icon: <Dataset fontSize="small" />,
    hint: 'Dataset versions (hashes) and training provenance links',
  },
] as const

export function AppLayout() {
  const location = useLocation()
  return (
    <Box>
      <AppBar position="sticky" elevation={0} sx={{ borderBottom: '1px solid #25314c' }}>
        <Toolbar sx={{ justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Trustworthy AI Control Center
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            {nav.map((item) => (
              <Tooltip key={item.to} title={item.hint} enterDelay={400}>
                <Button
                  component={NavLink}
                  to={item.to}
                  startIcon={item.icon}
                  variant={location.pathname === item.to ? 'contained' : 'text'}
                  color={location.pathname === item.to ? 'secondary' : 'inherit'}
                >
                  {item.label}
                </Button>
              </Tooltip>
            ))}
            <Button href="/docs" target="_blank" rel="noreferrer" variant="outlined" size="small">
              API docs
            </Button>
          </Box>
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ py: 3 }}>
        <Outlet />
      </Container>
      <HelpDrawer floating />
    </Box>
  )
}

