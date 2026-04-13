import { Analytics, Dataset, Hub, ModelTraining } from '@mui/icons-material'
import {
  AppBar,
  Box,
  Button,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { HelpDrawer } from './HelpDrawer'

const SIDEBAR_WIDTH = 220

const nav = [
  {
    label: 'Overview',
    to: '/',
    icon: <Analytics />,
    hint: 'KPIs, latest drift counts, run a drift check against the baseline',
  },
  {
    label: 'Workflows',
    to: '/workflows',
    icon: <Hub />,
    hint: 'History of orchestration runs and whether policy triggered',
  },
  {
    label: 'Models',
    to: '/models',
    icon: <ModelTraining />,
    hint: 'Retrain, promote stages, see experiments and production pointer',
  },
  {
    label: 'Data',
    to: '/data',
    icon: <Dataset />,
    hint: 'Dataset versions (hashes) and training provenance links',
  },
  {
    label: 'Inference',
    to: '/inference',
    icon: <Analytics />,
    hint: 'Run prediction against the current production model pointer',
  },
] as const

export function AppLayout() {
  const location = useLocation()
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* Left sidebar */}
      <Drawer
        variant="permanent"
        sx={{
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: SIDEBAR_WIDTH,
            boxSizing: 'border-box',
            bgcolor: '#f0f7ff',
            borderRight: '1.5px solid #bfdbfe',
            pt: 2,
          },
        }}
      >
        <Box sx={{ px: 2, pb: 2 }}>
          <Typography
            variant="subtitle1"
            sx={{
              fontWeight: 800,
              color: 'primary.main',
              lineHeight: 1.3,
              letterSpacing: '-0.02em',
            }}
          >
            Trustworthy AI
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Control Center
          </Typography>
        </Box>
        <List sx={{ px: 1 }}>
          {nav.map((item) => {
            const active = location.pathname === item.to
            return (
              <Tooltip key={item.to} title={item.hint} placement="right" enterDelay={500}>
                <ListItemButton
                  component={NavLink}
                  to={item.to}
                  sx={{
                    borderRadius: 3,
                    mb: 0.5,
                    bgcolor: active ? '#dbeafe' : 'transparent',
                    color: active ? 'primary.main' : 'text.primary',
                    '&:hover': { bgcolor: '#dbeafe' },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 36,
                      color: active ? 'primary.main' : 'text.secondary',
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.label}
                    slotProps={{
                      primary: {
                        sx: {
                          fontWeight: active ? 700 : 500,
                          fontSize: '0.9rem',
                        },
                      },
                    }}
                  />
                </ListItemButton>
              </Tooltip>
            )
          })}
        </List>
      </Drawer>

      {/* Main content area */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Top bar with API Docs on the right */}
        <AppBar
          position="sticky"
          elevation={0}
          sx={{
            bgcolor: '#ffffff',
            color: 'text.primary',
            borderBottom: '1.5px solid #bfdbfe',
          }}
        >
          <Toolbar sx={{ justifyContent: 'flex-end', minHeight: 52 }}>
            <Button
              href="/docs"
              target="_blank"
              rel="noreferrer"
              variant="outlined"
              size="small"
              sx={{
                borderColor: '#93c5fd',
                color: 'primary.main',
                fontWeight: 600,
                '&:hover': { borderColor: '#3b82f6', bgcolor: '#eff6ff' },
              }}
            >
              API Docs
            </Button>
          </Toolbar>
        </AppBar>

        <Box sx={{ flexGrow: 1, p: 3, maxWidth: 1400, mx: 'auto', width: '100%' }}>
          <Outlet />
        </Box>
      </Box>

      <HelpDrawer floating />
    </Box>
  )
}
