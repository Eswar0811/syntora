'use client';

import { useState, useEffect, useRef } from 'react';
import type { SpotifyCurrentResponse } from '../types';
import { getSessionId } from '../api';

interface Options {
  enabled: boolean;
  onUnauthorized: () => void;
  /** Called when the SSE connection dies permanently (triggers polling fallback) */
  onFallback: () => void;
}

const MAX_CONSECUTIVE_ERRORS = 3;

export function useSpotifySSE({ enabled, onUnauthorized, onFallback }: Options) {
  const [data, setData]       = useState<SpotifyCurrentResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const esRef              = useRef<EventSource | null>(null);
  const errorCountRef      = useRef(0);
  const onUnauthorizedRef  = useRef(onUnauthorized);
  const onFallbackRef      = useRef(onFallback);

  useEffect(() => { onUnauthorizedRef.current = onUnauthorized; }, [onUnauthorized]);
  useEffect(() => { onFallbackRef.current = onFallback; }, [onFallback]);

  useEffect(() => {
    if (!enabled) {
      esRef.current?.close();
      esRef.current = null;
      setData(null);
      setError(null);
      return;
    }

    // EventSource is browser-only
    if (typeof EventSource === 'undefined') {
      onFallbackRef.current();
      return;
    }

    setIsLoading(true);
    errorCountRef.current = 0;

    const sid = getSessionId();
    const es = new EventSource(`/api/spotify/stream${sid ? `?sid=${encodeURIComponent(sid)}` : ''}`);
    esRef.current = es;

    es.addEventListener('update', (e: MessageEvent) => {
      setIsLoading(false);
      errorCountRef.current = 0;
      try {
        setData(JSON.parse(e.data) as SpotifyCurrentResponse);
        setError(null);
      } catch { /* ignore malformed JSON */ }
    });

    es.addEventListener('ping', () => {
      setIsLoading(false);
      errorCountRef.current = 0;
    });

    es.addEventListener('auth_error', () => {
      es.close();
      onUnauthorizedRef.current();
    });

    es.addEventListener('error', (e: MessageEvent) => {
      try {
        const parsed = JSON.parse(e.data) as { message?: string };
        setError(parsed.message ?? 'Stream error');
      } catch { /* named error event with no parseable body */ }
    });

    /* onerror fires on network-level failures — EventSource auto-reconnects,
       but we count consecutive failures and fall back to polling if too many. */
    es.onerror = () => {
      setIsLoading(false);
      errorCountRef.current += 1;

      if (es.readyState === EventSource.CLOSED) {
        // Connection permanently closed — switch to polling
        onFallbackRef.current();
        return;
      }

      if (errorCountRef.current >= MAX_CONSECUTIVE_ERRORS) {
        es.close();
        onFallbackRef.current();
      } else {
        setError(`Stream error — reconnecting (${errorCountRef.current}/${MAX_CONSECUTIVE_ERRORS})`);
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [enabled]);

  return { data, error, isLoading };
}
