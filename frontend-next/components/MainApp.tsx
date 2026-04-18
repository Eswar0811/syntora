'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import type { AuthStatus } from '@/lib/types';
import { getSpotifyAuthUrl, spotifyLogout, markConnected, markDisconnected, isConnected } from '@/lib/api';
import LiveLyrics from './LiveLyrics';

export default function MainApp() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const [authStatus, setAuthStatus] = useState<AuthStatus>('idle');
  const [authError, setAuthError]   = useState<string | null>(null);

  useEffect(() => {
    if (isConnected()) setAuthStatus('authed');
  }, []);

  useEffect(() => {
    const spotify = searchParams.get('spotify');
    const reason  = searchParams.get('reason');
    if (spotify === 'connected') {
      markConnected();
      setAuthStatus('authed');
    } else if (spotify === 'error') {
      setAuthStatus('error');
      setAuthError(reason ?? 'Unknown error');
    }
    if (spotify) router.replace('/');
  }, [searchParams, router]);

  const handleConnect = useCallback(async () => {
    setAuthStatus('exchanging');
    setAuthError(null);
    try {
      const url = await getSpotifyAuthUrl();
      window.location.href = url;
    } catch {
      setAuthStatus('error');
      setAuthError('Could not reach the server — is the backend running?');
    }
  }, []);

  const handleDisconnect = useCallback(async () => {
    await spotifyLogout();
    markDisconnected();
    setAuthStatus('idle');
    setAuthError(null);
  }, []);

  return (
    <main className="min-h-dvh bg-[#121212]">
      <div className="mx-auto max-w-2xl px-4 pb-16 pt-8 sm:px-6">
        <header className="mb-8 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/syntora-logo.png" alt="Syntora" className="h-10 w-10 rounded-xl object-cover" />
            <div>
              <span className="text-gradient-brand text-xl font-extrabold tracking-tight">Syntora</span>
              <p className="text-xs text-[#b3b3b3]">Your Song, Your Vibe!</p>
            </div>
          </div>
          {authStatus === 'authed' && (
            <button
              type="button"
              onClick={handleDisconnect}
              className="rounded-full border border-white/10 px-4 py-1.5 text-xs font-semibold
                         text-[#b3b3b3] transition hover:border-white/20 hover:text-white"
            >
              Disconnect
            </button>
          )}
        </header>

        <LiveLyrics
          authStatus={authStatus}
          authError={authError}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
        />

        <footer className="mt-12 text-center text-xs font-bold tracking-widest uppercase text-[#b3b3b3]/40">
          Syntora
        </footer>
      </div>
    </main>
  );
}
