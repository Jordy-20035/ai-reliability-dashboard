import { Card, CardContent, Typography } from '@mui/material'

interface Props {
  label: string
  value: string | number
  subtitle?: string
}

export function KpiCard({ label, value, subtitle }: Props) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography color="text.secondary" variant="body2">
          {label}
        </Typography>
        <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>
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

