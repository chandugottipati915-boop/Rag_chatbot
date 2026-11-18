import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `npm run build` writes plain HTML/JS/CSS to out/. The API serves those files itself
  // (see the Dockerfile), so production is one service on one URL.
  output: "export",
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
