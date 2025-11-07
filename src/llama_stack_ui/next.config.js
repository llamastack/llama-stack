/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    // TODO: Remove this once we fix the build errors
    ignoreBuildErrors: true,
  },
  output: 'standalone',
};

module.exports = nextConfig;
