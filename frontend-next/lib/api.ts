const API = '/api';

const SESSION_KEY = 'syntora_sid';

export function getSessionId(): string {
  return (typeof window !== 'undefined' && localStorage.getItem(SESSION_KEY)) ?? '';
}

export function setSessionId(sid: string): void {
  if (typeof window !== 'undefined') localStorage.setItem(SESSION_KEY, sid);
}

export function clearSessionId(): void {
  if (typeof window !== 'undefined') localStorage.removeItem(SESSION_KEY);
}

export function sessionHeaders(): Record<string, string> {
  const sid = getSessionId();
  return sid ? { 'X-Session-ID': sid } : {};
}

export async function getSpotifyAuthUrl(): Promise<string> {
  const res = await fetch(`${API}/spotify/auth-url`);
  if (!res.ok) throw new Error(`Auth URL fetch failed: ${res.status}`);
  const data = await res.json();
  return data.url as string;
}

export async function spotifyLogout(): Promise<void> {
  await fetch(`${API}/spotify/logout`, {
    method: 'POST',
    headers: sessionHeaders(),
  });
  clearSessionId();
}

export async function healthCheck(): Promise<Record<string, boolean>> {
  const res = await fetch(`${API}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
