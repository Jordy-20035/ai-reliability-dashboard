import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#3b82f6' },
    secondary: { main: '#2563eb' },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    text: {
      primary: '#1e293b',
      secondary: '#64748b',
    },
    divider: '#bfdbfe',
  },
  shape: { borderRadius: 18 },
  components: {
    MuiCard: {
      defaultProps: { variant: 'outlined' },
      styleOverrides: {
        root: {
          borderColor: '#93c5fd',
          borderWidth: 1.5,
          borderRadius: 20,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        outlined: {
          borderColor: '#93c5fd',
          borderWidth: 1.5,
          borderRadius: 20,
        },
        root: {
          borderRadius: 20,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 14,
          textTransform: 'none' as const,
          fontWeight: 600,
        },
        outlined: {
          borderColor: '#93c5fd',
          '&:hover': { borderColor: '#3b82f6', backgroundColor: '#eff6ff' },
        },
        contained: {
          boxShadow: 'none',
          '&:hover': { boxShadow: '0 2px 8px rgba(59,130,246,0.25)' },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: { borderRadius: 14 },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 14,
          '& .MuiOutlinedInput-notchedOutline': {
            borderColor: '#93c5fd',
          },
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: '#3b82f6',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff',
          color: '#1e293b',
          borderBottom: '1.5px solid #bfdbfe',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          borderRight: '1.5px solid #bfdbfe',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 16,
        },
      },
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ThemeProvider>
  </StrictMode>,
)
