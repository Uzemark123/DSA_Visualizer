/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#020617',
        panel: '#0b1224',
        card: '#0f172a',
        accent: {
          teal: '#22d3ee',
        },
      },
      borderRadius: {
        xl2: '1.25rem',
      },
      boxShadow: {
        float: '0 18px 50px rgba(0,0,0,0.35)',
      },
    },
  },
  plugins: [],
}
