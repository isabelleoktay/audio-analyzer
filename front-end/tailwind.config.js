/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        chakra: ["Chakra Petch", "sans-serif"],
        poppins: ["Poppins", "sans-serif"],
      },
      backgroundImage: {
        radial: "radial-gradient(var(--tw-gradient-stops))",
      },
      colors: {
        blueblack: "#1E1E2F",
        bluegray: "#5F5F95",
        electricblue: "#90F1EF",
        warmyellow: "#FFCB6B",
        lightgray: "#E0E0E0",
        lightpink: "#FFD6E8",
        darkpink: "#FF89BB",
      },
    },
  },
  plugins: [],
};
