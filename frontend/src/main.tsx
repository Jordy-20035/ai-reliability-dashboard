import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#1a5fb4' },
    secondary: { main: '#2563eb' },
    success: { main: '#16a34a' },
    warning: { main: '#d97706' },
    error: { main: '#dc2626' },
    background: {
      default: '#f4f6f9',
      paper: '#ffffff',
    },
    text: {
      primary: '#1e293b',
      secondary: '#64748b',
    },
    divider: '#e2e8f0',
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", Roboto, Arial, sans-serif',
    h4: { fontWeight: 700, fontSize: '1.5rem' },
    h6: { fontWeight: 700, fontSize: '1.05rem' },
  },
  shape: { borderRadius: 14 },
  components: {
    MuiCard: {
      defaultProps: { variant: 'outlined' },
      styleOverrides: {
        root: {
          borderColor: '#e2e8f0',
          borderWidth: 1,
          borderRadius: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        outlined: {
          borderColor: '#e2e8f0',
          borderWidth: 1,
          borderRadius: 16,
        },
        root: {
          borderRadius: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          textTransform: 'none' as const,
          fontWeight: 600,
          fontSize: '0.85rem',
        },
        outlined: {
          borderColor: '#cbd5e1',
          '&:hover': { borderColor: '#1a5fb4', backgroundColor: '#f0f4ff' },
        },
        contained: {
          boxShadow: 'none',
          '&:hover': { boxShadow: '0 2px 8px rgba(26,95,180,0.25)' },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: { borderRadius: 10 },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          '& .MuiOutlinedInput-notchedOutline': {
            borderColor: '#cbd5e1',
          },
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: '#1a5fb4',
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: { borderRadius: 12 },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { fontWeight: 600, fontSize: '0.75rem' },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: { borderRadius: 8, height: 3 },
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
