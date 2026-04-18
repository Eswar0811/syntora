'use client';

/**
 * useSpotifyLive — unified live data hook.
 *
 * Strategy:
 *   1. Attempts Server-Sent Events via the Edge Runtime proxy route.
 *   2. On 3 consecutive SSE errors or a permanent close, seamlessly falls back
 *      to the adaptive-interval polling hook.
 *
 * Both branches return the same { data, error, isLoading } shape, so the
 * consuming component doesn't need to know which transport is active.
 */

import { useState, useCallback } from 'react';
import type { SpotifyCurrentResponse } from '../types';
import { useSpotifySSE } from './useSpotifySSE';
import { useSpotifyPoll } from './useSpotifyPoll';

interface Options {
  enabled: boolean;
  onUnauthorized: () => void;
}

interface LiveState {
  data: SpotifyCurrentResponse | null;
  error: string | null;
  isLoading: boolean;
  transport: 'sse' | 'poll';
}

export function useSpotifyLive({ enabled, onUnauthorized }: Options): LiveState {
  const [usePoll, setUsePoll] = useState(false);

  const handleFallback = useCallback(() => setUsePoll(true), []);

  const sse = useSpotifySSE({
    enabled: enabled && !usePoll,
    onUnauthorized,
    onFallback: handleFallback,
  });

  const poll = useSpotifyPoll({
    enabled: enabled && usePoll,
    onUnauthorized,
  });

  if (usePoll) return { ...poll, transport: 'poll' };
  return { ...sse, transport: 'sse' };
}
