const API = '/api';
const CONNECTED_KEY = 'syntora_connected';
const SID_KEY       = 'syntora_sid';

// ── Session helpers ───────────────────────────────────────────────────────────

export function isConnected(): boolean {
  if (typeof window === 'undefined') return false;
  return localStorage.getItem(CONNECTED_KEY) === 'true' && Boolean(localStorage.getItem(SID_KEY));
}

export function markConnected(sid: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(CONNECTED_KEY, 'true');
  localStorage.setItem(SID_KEY, sid);
}

export function markDisconnected(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(CONNECTED_KEY);
  localStorage.removeItem(SID_KEY);
}

function getSid(): string {
  return (typeof window !== 'undefined' && localStorage.getItem(SID_KEY)) || '';
}

// Appends ?sid=<session-id> so it survives Vercel → Render proxy (headers get stripped)
export function apiFetch(path: string, opts: RequestInit = {}): Promise<Response> {
  const sid = getSid();
  const sep = path.includes('?') ? '&' : '?';
  const url = sid ? `${API}${path}${sep}sid=${encodeURIComponent(sid)}` : `${API}${path}`;
  return fetch(url, opts);
}

// ── Public API helpers ────────────────────────────────────────────────────────

export async function getSpotifyAuthUrl(): Promise<string> {
  const res = await apiFetch('/spotify/auth-url');
  if (!res.ok) throw new Error(`Auth URL fetch failed: ${res.status}`);
  const data = await res.json();
  return data.url as string;
}

export async function spotifyLogout(): Promise<void> {
  await apiFetch('/spotify/logout', { method: 'POST' }).catch(() => {});
  markDisconnected();
}

export async function healthCheck(): Promise<Record<string, unknown>> {
  const res = await apiFetch('/health');
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
