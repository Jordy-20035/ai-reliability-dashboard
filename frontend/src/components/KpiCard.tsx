import { Card, CardContent, Typography } from '@mui/material'

interface Props {
  label: string
  value: string | number
  subtitle?: string
}

export function KpiCard({ label, value, subtitle }: Props) {
  return (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        borderColor: '#93c5fd',
        borderWidth: 1.5,
        borderRadius: 5,
        transition: 'box-shadow 0.2s, border-color 0.2s',
        '&:hover': {
          borderColor: '#3b82f6',
          boxShadow: '0 4px 16px rgba(59,130,246,0.12)',
        },
      }}
    >
      <CardContent>
        <Typography color="text.secondary" variant="body2" sx={{ fontWeight: 500 }}>
          {label}
        </Typography>
        <Typography variant="h4" sx={{ mt: 1, fontWeight: 700, color: 'primary.main' }}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  )
}
