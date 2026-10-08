import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      // Proxy API requests to FastAPI backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy SSE stream requests
      '/api/stream': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // SSE needs these headers
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            proxyReq.setHeader('Accept', 'text/event-stream')
          })
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
