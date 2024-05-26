const withMT = require("@material-tailwind/react/utils/withMT");

module.exports = withMT({
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    colors:{
        "ivory":"#FFFFF0"
    },
      extend: {
    // Import external CSS file for fonts
   
  },
  },
  plugins: [require("daisyui")],
});
