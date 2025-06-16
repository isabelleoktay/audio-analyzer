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
      borderDasharray: {
        "8-4": "8 4", // Dash length 8px, gap 4px
        "10-5": "10 5", // Dash length 10px, gap 5px
      },
    },
  },
  plugins: [
    function ({ addUtilities }) {
      addUtilities({
        ".border-dash-8-4": {
          borderStyle: "dashed",
          borderDasharray: "8 4",
        },
        ".border-dash-10-5": {
          borderStyle: "dashed",
          borderDasharray: "10 5",
        },
      });
    },
  ],
};
