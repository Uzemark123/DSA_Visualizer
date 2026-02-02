/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#18181B',
        panel: '#27272A',
        card: '#27272A',
        accent: {
          amber: '#F59E0B',
          orange: '#FB923C',
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
