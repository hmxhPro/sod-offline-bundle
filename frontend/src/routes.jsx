/**
 * src/routes.jsx
 * --------------
 * Central route table. A single layout route (ConsoleLayout — the persistent
 * shell + state host) wraps the four pages, plus a catch-all that redirects
 * unknown paths home.
 */

import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import ConsoleLayout from './layout/ConsoleLayout'
import DashboardPage from './pages/DashboardPage'
import DetectPage from './pages/DetectPage'
import ImageDetectPage from './pages/ImageDetectPage'
import TrainingPage from './pages/TrainingPage'
import HistoryPage from './pages/HistoryPage'
import LLMAnnotationPage from './pages/LLMAnnotationPage'

export default function AppRoutes() {
  return (
    <Routes>
      <Route element={<ConsoleLayout />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/detect" element={<DetectPage />} />
        <Route path="/image-detect" element={<ImageDetectPage />} />
        <Route path="/training" element={<TrainingPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/llm-annotation" element={<LLMAnnotationPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
