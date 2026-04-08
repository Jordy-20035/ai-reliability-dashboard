import { Analytics, Dataset, Hub, ModelTraining } from '@mui/icons-material'
import {
  AppBar,
  Box,
  Button,
  Container,
  Toolbar,
  Typography,
} from '@mui/material'
import { NavLink, Outlet, useLocation } from 'react-router-dom'

const nav = [
  { label: 'Overview', to: '/', icon: <Analytics fontSize="small" /> },
  { label: 'Workflows', to: '/workflows', icon: <Hub fontSize="small" /> },
  { label: 'Models', to: '/models', icon: <ModelTraining fontSize="small" /> },
  { label: 'Data', to: '/data', icon: <Dataset fontSize="small" /> },
]

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
              <Button
                key={item.to}
                component={NavLink}
                to={item.to}
                startIcon={item.icon}
                variant={location.pathname === item.to ? 'contained' : 'text'}
                color={location.pathname === item.to ? 'secondary' : 'inherit'}
              >
                {item.label}
              </Button>
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
    </Box>
  )
}

