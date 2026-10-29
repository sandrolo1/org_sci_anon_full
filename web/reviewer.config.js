import { defineConfig } from 'vite';

export default defineConfig({
  root: 'reviewer',
  base: './',
  build: { outDir: 'dist-reviewer', emptyOutDir: true },
});
