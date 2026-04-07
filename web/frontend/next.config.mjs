import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  outputFileTracingRoot: path.join(__dirname),
  async rewrites() {
    // In production (Vercel), API_URL env var points to your home machine
    // In development, falls back to localhost
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
      {
        // WebSocket rewrite - must use http:// in config, browser handles upgrade
        source: '/ws',
        destination: `${apiUrl}/ws`,
      },
    ]
  },
};

export default nextConfig;
