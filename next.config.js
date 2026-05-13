/** @type {import('next').NextConfig} */

let withBundleAnalyzer = require("@next/bundle-analyzer")({
    enabled: process.env.ANALYZE === "true",
});

const nextConfig = {
  reactStrictMode: false, // Recommended for the `pages` directory, default in `app`.
  productionBrowserSourceMaps: true,
  outputFileTracingRoot: __dirname,
  redirects: async () => {
    return [
      {
        source: "/llm-viz",
        destination: "/llm",
        permanent: true,
      },
    ];
  }
};

module.exports = withBundleAnalyzer(nextConfig);
