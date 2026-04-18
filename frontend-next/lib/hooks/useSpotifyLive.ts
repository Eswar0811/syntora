'use client';

import type { SpotifyCurrentResponse } from '../types';
import { useSpotifyPoll } from './useSpotifyPoll';

interface Options {
  enabled: boolean;
  onUnauthorized: () => void;
}

interface LiveState {
  data: SpotifyCurrentResponse | null;
  error: string | null;
  isLoading: boolean;
  transport: 'poll';
}

// Use adaptive polling as the sole transport.
// SSE via Vercel Edge has a 25-second streaming timeout on the free tier
// which causes constant reconnection loops. Polling at 1s intervals when
// a song is playing is imperceptible to the user and fully reliable.
export function useSpotifyLive({ enabled, onUnauthorized }: Options): LiveState {
  const poll = useSpotifyPoll({ enabled, onUnauthorized });
  return { ...poll, transport: 'poll' };
}
