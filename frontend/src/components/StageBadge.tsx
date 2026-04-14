import { Chip } from '@mui/material'

const colors: Record<string, 'success' | 'primary' | 'warning' | 'default' | 'error'> = {
  production: 'success',
  staging: 'primary',
  development: 'warning',
  archived: 'default',
}

export function StageBadge({ stage }: { stage: string }) {
  const color = colors[stage] ?? 'default'
  return <Chip label={stage} size="small" color={color} variant="outlined" />
}
