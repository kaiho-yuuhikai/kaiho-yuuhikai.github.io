import type { Config } from 'tailwindcss'

export default {
  content: [
    './components/**/*.{js,vue,ts}',
    './layouts/**/*.vue',
    './pages/**/*.vue',
    './plugins/**/*.{js,ts}',
    './app.vue',
    './error.vue'
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Noto Sans JP"', 'system-ui', 'sans-serif'],
      },
      colors: {
        kaiho: {
          green: '#2d8c3c',
          'green-dark': '#1f6b2a',
          cream: '#f5f1e8',
          'cream-dark': '#e8e2d4',
          gold: '#d4a017',
        }
      },
    },
  },
  plugins: [],
} satisfies Config
