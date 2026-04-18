const API = '/api';

export async function getSpotifyAuthUrl(): Promise<string> {
  const res = await fetch(`${API}/spotify/auth-url`);
  if (!res.ok) throw new Error(`Auth URL fetch failed: ${res.status}`);
  const data = await res.json();
  return data.url as string;
}

export async function spotifyLogout(): Promise<void> {
  await fetch(`${API}/spotify/logout`, { method: 'POST' });
}

export async function healthCheck(): Promise<Record<string, boolean>> {
  const res = await fetch(`${API}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
