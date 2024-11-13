/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Chakra Petch", "sans-serif"], // Add your new font here
      },
    },
  },
  plugins: [],
};
