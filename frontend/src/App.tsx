import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { DataPage } from './pages/DataPage'
import { InferencePage } from './pages/InferencePage'
import { ModelsPage } from './pages/ModelsPage'
import { OverviewPage } from './pages/OverviewPage'
import { WorkflowsPage } from './pages/WorkflowsPage'

function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/workflows" element={<WorkflowsPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/data" element={<DataPage />} />
        <Route path="/inference" element={<InferencePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default App
