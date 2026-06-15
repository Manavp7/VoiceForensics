/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Optionally proxy API calls to the backend during local dev to avoid CORS.
  async rewrites() {
    const target = process.env.API_PROXY_TARGET;
    if (!target) return [];
    return [{ source: "/v1/:path*", destination: `${target}/v1/:path*` }];
  },
};

export default nextConfig;
