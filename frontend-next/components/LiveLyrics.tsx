'use client';

import { useCallback } from 'react';
import type { AuthStatus } from '@/lib/types';
import { useSpotifyLive } from '@/lib/hooks/useSpotifyLive';
import ConnectCard from './ConnectCard';
import NowPlaying from './NowPlaying';
import IdleCard from './IdleCard';

interface Props {
  authStatus: AuthStatus;
  authError: string | null;
  onConnect: () => void;
  onDisconnect: () => void;
}

export default function LiveLyrics({ authStatus, authError, onConnect, onDisconnect }: Props) {
  const handleUnauthorized = useCallback(() => onDisconnect(), [onDisconnect]);

  const { data, error, isLoading, transport } = useSpotifyLive({
    enabled: authStatus === 'authed',
    onUnauthorized: handleUnauthorized,
  });

  /* ── Unauthenticated states ── */
  if (authStatus !== 'authed') {
    return <ConnectCard status={authStatus} error={authError} onConnect={onConnect} />;
  }

  /* ── First-connect loading ── */
  if (isLoading && !data) {
    return (
      <div className="card-glass p-8 text-center">
        <div className="mb-4 flex justify-center">
          <Spinner />
        </div>
        <p className="text-sm text-[#b3b3b3]">Connecting to Spotify…</p>
      </div>
    );
  }

  /* ── Hard error with no data yet ── */
  if (error && !data) {
    return (
      <div className="card-glass p-8 text-center">
        <p className="mb-2 text-sm font-semibold text-red-400">Connection error</p>
        <p className="text-xs text-[#b3b3b3]">{error}</p>
      </div>
    );
  }

  /* ── Nothing playing ── */
  if (!data?.is_playing) {
    return <IdleCard error={error} />;
  }

  /* ── Now playing ── */
  return <NowPlaying data={data} transport={transport} />;
}

function Spinner() {
  return (
    <svg
      className="h-8 w-8 animate-spin text-[#1db954]"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" strokeOpacity={0.2} />
      <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
    </svg>
  );
}
