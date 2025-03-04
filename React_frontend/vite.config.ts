import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react' // If you are using React

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()], // Add react plugin if you are using React
})