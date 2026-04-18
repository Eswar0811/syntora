import type { NextConfig } from 'next';

const backendUrl = process.env.BACKEND_URL ?? 'http://localhost:8000';

const nextConfig: NextConfig = {
  // ── API proxy ─────────────────────────────────────────────────────────────
  // All /api/* requests are forwarded to the FastAPI backend EXCEPT
  // /api/spotify/stream which is handled by the Edge Runtime route in
  // app/api/spotify/stream/route.ts (file-based routes take priority).
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`,
      },
    ];
  },

  // ── Security headers ──────────────────────────────────────────────────────
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Frame-Options',           value: 'DENY' },
          { key: 'X-Content-Type-Options',     value: 'nosniff' },
          { key: 'Referrer-Policy',            value: 'strict-origin-when-cross-origin' },
          { key: 'Permissions-Policy',         value: 'camera=(), microphone=(), geolocation=()' },
          { key: 'X-DNS-Prefetch-Control',     value: 'on' },
        ],
      },
      // Allow SSE stream to bypass any intermediary caching
      {
        source: '/api/spotify/stream',
        headers: [
          { key: 'Cache-Control',    value: 'no-cache, no-transform' },
          { key: 'X-Accel-Buffering', value: 'no' },
        ],
      },
    ];
  },

  // ── Build optimisations ───────────────────────────────────────────────────
  compress: true,          // gzip/br at the Next.js layer (Vercel does this anyway)
  poweredByHeader: false,  // don't leak the Next.js version

  // Image optimisation — keep defaults; extend when album art is added
  images: {
    formats: ['image/avif', 'image/webp'],
    minimumCacheTTL: 3600,
  },

  // ── Logging ───────────────────────────────────────────────────────────────
  logging: {
    fetches: {
      fullUrl: process.env.NODE_ENV === 'development',
    },
  },
};

export default nextConfig;
