import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        spotify: {
          green: '#1db954',
          'green-hover': '#1ed760',
          black: '#121212',
          dark: '#181818',
          surface: '#282828',
          surface2: '#333333',
          muted: '#b3b3b3',
        },
        lang: {
          tamil: '#c084fc',
          'tamil-dim': '#7c3aed',
          hindi: '#38bdf8',
          'hindi-dim': '#0284c7',
          malayalam: '#34d399',
          'malayalam-dim': '#059669',
          telugu: '#fb923c',
          'telugu-dim': '#ea580c',
        },
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', '-apple-system', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.25s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-dot': 'pulseDot 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pulseDot: {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.5', transform: 'scale(0.85)' },
        },
      },
      boxShadow: {
        card: '0 4px 24px rgba(0,0,0,0.4)',
        glow: '0 0 20px rgba(29, 185, 84, 0.3)',
        'glow-tamil': '0 0 20px rgba(192, 132, 252, 0.25)',
        'glow-hindi': '0 0 20px rgba(56, 189, 248, 0.25)',
        'glow-malayalam': '0 0 20px rgba(52, 211, 153, 0.25)',
        'glow-telugu': '0 0 20px rgba(251, 146, 60, 0.25)',
      },
    },
  },
  plugins: [],
};

export default config;
