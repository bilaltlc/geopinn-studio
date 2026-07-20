import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: './',
  
  server: { hmr: { host: 'localhost', port: 5173 } },
  plugins: [
    tailwindcss(),
    react(),
  ],
  build: {
    cssMinify: 'esbuild',   // lightningcss yerine
    // SourceMap'leri kapatmak build belleğini ciddi ölçüde rahatlatır
    sourcemap: false,
    rollupOptions: {
      output: {
        // Ağır kütüphaneleri (Plotly, Three) ayrı chunk'lara al
        manualChunks(id) {
          if (id.includes('plotly.js-gl3d-dist-min') || id.includes('react-plotly.js')) {
            return 'plotly'
          }
          if (id.includes('three') || id.includes('@react-three')) {
            return 'three'
          }
        },
      },
    },
  },
})