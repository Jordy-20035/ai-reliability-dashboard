import {
  Analytics,
  Dataset,
  Hub,
  ModelTraining,
  OpenInNew,
} from '@mui/icons-material'
import {
  AppBar,
  Avatar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Stack,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { HelpDrawer } from './HelpDrawer'

const SIDEBAR_WIDTH = 210

const nav = [
  {
    label: 'Overview',
    to: '/',
    icon: <Analytics fontSize="small" />,
    hint: 'KPIs, latest drift counts, run a drift check',
  },
  {
    label: 'Workflows',
    to: '/workflows',
    icon: <Hub fontSize="small" />,
    hint: 'History of orchestration runs',
  },
  {
    label: 'Models',
    to: '/models',
    icon: <ModelTraining fontSize="small" />,
    hint: 'Registry, retrain, promote, lifecycle',
  },
  {
    label: 'Data',
    to: '/data',
    icon: <Dataset fontSize="small" />,
    hint: 'Dataset versions and provenance',
  },
  {
    label: 'Inference',
    to: '/inference',
    icon: <Analytics fontSize="small" />,
    hint: 'Score against production model',
  },
] as const

export function AppLayout() {
  const location = useLocation()
  const activeLabel = nav.find((n) => n.to === location.pathname)?.label ?? 'Overview'

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* ---- LEFT SIDEBAR ---- */}
      <Drawer
        variant="permanent"
        sx={{
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: SIDEBAR_WIDTH,
            boxSizing: 'border-box',
            bgcolor: '#ffffff',
            borderRight: '1px solid #e2e8f0',
            display: 'flex',
            flexDirection: 'column',
          },
        }}
      >
        <Box sx={{ px: 2.5, pt: 2.5, pb: 1.5 }}>
          <Stack direction="row" sx={{ alignItems: 'center' }} spacing={1}>
            <Avatar
              sx={{
                bgcolor: '#1a5fb4',
                width: 32,
                height: 32,
                fontSize: 14,
                fontWeight: 800,
              }}
            >
              AI
            </Avatar>
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                Trustworthy AI
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                Control Center
              </Typography>
            </Box>
          </Stack>
        </Box>

        <List sx={{ px: 1, flexGrow: 1 }}>
          {nav.map((item) => {
            const active = location.pathname === item.to
            return (
              <Tooltip key={item.to} title={item.hint} placement="right" enterDelay={600}>
                <ListItemButton
                  component={NavLink}
                  to={item.to}
                  sx={{
                    borderRadius: 2.5,
                    mb: 0.25,
                    py: 0.75,
                    bgcolor: active ? '#eef2ff' : 'transparent',
                    color: active ? 'primary.main' : 'text.primary',
                    '&:hover': { bgcolor: active ? '#eef2ff' : '#f8fafc' },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 32,
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
                          fontSize: '0.84rem',
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

      {/* ---- MAIN ---- */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Top bar */}
        <AppBar
          position="sticky"
          elevation={0}
          sx={{
            bgcolor: '#ffffff',
            color: 'text.primary',
            borderBottom: '1px solid #e2e8f0',
            zIndex: (t) => t.zIndex.drawer - 1,
          }}
        >
          <Toolbar sx={{ justifyContent: 'space-between', minHeight: 52 }}>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {activeLabel}
            </Typography>

            <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
              <Tooltip title="OpenAPI docs (new tab)">
                <IconButton
                  href="/docs"
                  target="_blank"
                  rel="noreferrer"
                  size="small"
                  sx={{ color: 'text.secondary' }}
                >
                  <OpenInNew fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          </Toolbar>
        </AppBar>

        <Box
          sx={{
            flexGrow: 1,
            p: { xs: 2, md: 3 },
            maxWidth: 1360,
            width: '100%',
            mx: 'auto',
          }}
        >
          <Outlet />
        </Box>
      </Box>

      <HelpDrawer floating />
    </Box>
  )
}
