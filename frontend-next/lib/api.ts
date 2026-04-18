const API = '/api';
const CONNECTED_KEY = 'syntora_connected';

export function isConnected(): boolean {
  return typeof window !== 'undefined' && localStorage.getItem(CONNECTED_KEY) === 'true';
}

export function markConnected(): void {
  if (typeof window !== 'undefined') localStorage.setItem(CONNECTED_KEY, 'true');
}

export function markDisconnected(): void {
  if (typeof window !== 'undefined') localStorage.removeItem(CONNECTED_KEY);
}

export async function getSpotifyAuthUrl(): Promise<string> {
  const res = await fetch(`${API}/spotify/auth-url`);
  if (!res.ok) throw new Error(`Auth URL fetch failed: ${res.status}`);
  const data = await res.json();
  return data.url as string;
}

export async function spotifyLogout(): Promise<void> {
  await fetch(`${API}/spotify/logout`, { method: 'POST' });
  markDisconnected();
}

export async function healthCheck(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
