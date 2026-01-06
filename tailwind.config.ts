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
        accent: '#1a1a1a',
      },
      letterSpacing: {
        wider: '0.1em',
        widest: '0.2em',
      }
    },
  },
  plugins: [],
} satisfies Config
