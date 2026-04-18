/**
 * Edge Runtime SSE proxy.
 *
 * Pipes the backend /spotify/stream SSE endpoint through to the browser.
 * Edge Runtime has no function timeout (unlike Serverless), so long-lived
 * SSE connections work correctly on Vercel's free tier.
 *
 * The Next.js /api/* rewrite in next.config.ts intentionally excludes this
 * path — file-based routes always take priority over rewrites.
 */
import { type NextRequest } from 'next/server';

export const runtime = 'edge';
export const dynamic = 'force-dynamic';

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function GET(req: NextRequest) {
  let upstream: Response;

  try {
    upstream = await fetch(`${BACKEND}/spotify/stream`, {
      headers: {
        Accept: 'text/event-stream',
        'Cache-Control': 'no-cache',
        // Forward the real client IP so the backend rate-limiter sees it
        'X-Forwarded-For': req.headers.get('x-forwarded-for') ?? '',
      },
      // Edge fetch supports streaming body — no need for any extra flags
    });
  } catch {
    // Backend unreachable — send a single SSE error event and close
    const body = `event: error\ndata: ${JSON.stringify({ message: 'Backend unreachable' })}\n\n`;
    return new Response(body, {
      status: 200,
      headers: sseHeaders(),
    });
  }

  if (!upstream.ok || !upstream.body) {
    const body = `event: error\ndata: ${JSON.stringify({ message: `Backend error ${upstream.status}` })}\n\n`;
    return new Response(body, {
      status: 200,
      headers: sseHeaders(),
    });
  }

  // Stream the upstream body directly — zero buffering
  return new Response(upstream.body, {
    status: 200,
    headers: sseHeaders(),
  });
}

function sseHeaders(): HeadersInit {
  return {
    'Content-Type':     'text/event-stream; charset=utf-8',
    'Cache-Control':    'no-cache, no-transform',
    'Connection':       'keep-alive',
    'X-Accel-Buffering': 'no',
  };
}
