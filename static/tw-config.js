// Tailwind Play CDN config — must load BEFORE the tailwind CDN script
window.tailwind = window.tailwind || {};
window.tailwind.config = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: 'rgb(var(--bg) / <alpha-value>)',
        surface: 'rgb(var(--surface) / <alpha-value>)',
        surface2: 'rgb(var(--surface-2) / <alpha-value>)',
        border: 'rgb(var(--border) / <alpha-value>)',
        text: 'rgb(var(--text) / <alpha-value>)',
        muted: 'rgb(var(--muted) / <alpha-value>)',
        subtle: 'rgb(var(--subtle) / <alpha-value>)',
        brand: {
          DEFAULT: 'rgb(var(--brand) / <alpha-value>)',
          dark: 'rgb(var(--brand-dark) / <alpha-value>)',
          tint: 'rgb(var(--brand-tint) / <alpha-value>)',
          ink: 'rgb(var(--brand-ink) / <alpha-value>)',
        },
        success: 'rgb(var(--success) / <alpha-value>)',
        warn: 'rgb(var(--warn) / <alpha-value>)',
        danger: 'rgb(var(--danger) / <alpha-value>)',
        rose: 'rgb(var(--rose) / <alpha-value>)',
        protein: 'rgb(var(--success) / <alpha-value>)',
        carbs: 'rgb(var(--warn) / <alpha-value>)',
        fat: 'rgb(var(--rose) / <alpha-value>)',
      },
      fontFamily: {
        sans: ['Geist', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['Fraunces', 'Georgia', 'serif'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        soft: '0 1px 2px rgba(20,15,10,0.04), 0 8px 24px rgba(20,15,10,0.05)',
        lift: '0 4px 12px rgba(20,15,10,0.06), 0 16px 40px rgba(20,15,10,0.08)',
      },
      letterSpacing: {
        widest2: '0.18em',
      },
    },
  },
};
