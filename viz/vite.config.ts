import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// Local-first single-page app. Base is relative ('./') so a static `dist/`
// build can be opened from any subpath or a simple file host when we make it
// deployable later. See viz/docs/01_architecture.md.
export default defineConfig({
  base: './',
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    strictPort: false,
    open: false,
  },
});
