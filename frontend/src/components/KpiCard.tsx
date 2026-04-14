import { Box, Card, CardContent, Typography } from '@mui/material'
import type { ReactNode } from 'react'

interface Props {
  label: string
  value: string | number
  subtitle?: string
  icon?: ReactNode
}

export function KpiCard({ label, value, subtitle, icon }: Props) {
  return (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        borderColor: '#e2e8f0',
        transition: 'box-shadow 0.2s',
        '&:hover': { boxShadow: '0 4px 12px rgba(0,0,0,0.06)' },
      }}
    >
      <CardContent sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
        {icon && (
          <Box
            sx={{
              mt: 0.25,
              width: 38,
              height: 38,
              borderRadius: 2.5,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: '#eef2ff',
              color: 'primary.main',
              flexShrink: 0,
            }}
          >
            {icon}
          </Box>
        )}
        <Box>
          <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 500 }}>
            {label}
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 700, lineHeight: 1.2, mt: 0.25 }}>
            {value}
          </Typography>
          {subtitle && (
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
              {subtitle}
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  )
}
