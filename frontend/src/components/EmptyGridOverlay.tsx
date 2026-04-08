import { Box, Typography } from '@mui/material'

export function EmptyGridOverlay({ message }: { message: string }) {
  return (
    <Box sx={{ py: 4, px: 2, textAlign: 'center' }}>
      <Typography variant="body2" color="text.secondary">
        {message}
      </Typography>
    </Box>
  )
}
