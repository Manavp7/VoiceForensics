import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0d1117",
        panel: "#161b22",
        accent: "#2f81f7",
      },
    },
  },
  plugins: [],
};

export default config;
