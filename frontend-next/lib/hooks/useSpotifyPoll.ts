'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type { SpotifyCurrentResponse } from '../types';
import { sessionHeaders } from '../api';

interface UseSpotifyPollOptions {
  enabled: boolean;
  onUnauthorized: () => void;
}

export function useSpotifyPoll({ enabled, onUnauthorized }: UseSpotifyPollOptions) {
  const [data, setData] = useState<SpotifyCurrentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const enabledRef = useRef(enabled);
  const inflightRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const onUnauthorizedRef = useRef(onUnauthorized);

  useEffect(() => { enabledRef.current = enabled; }, [enabled]);
  useEffect(() => { onUnauthorizedRef.current = onUnauthorized; }, [onUnauthorized]);

  const doPoll = useCallback(async () => {
    if (!enabledRef.current || inflightRef.current) return;

    inflightRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;

    let nextDelay = 3000;

    try {
      const res = await fetch('/api/spotify/current', {
        signal: controller.signal,
        headers: sessionHeaders(),
      });

      if (res.status === 401) {
        onUnauthorizedRef.current();
        return;
      }

      if (res.status === 429) {
        nextDelay = 12000;
        setError('Rate limited — slowing down');
        return;
      }

      if (!res.ok) {
        nextDelay = 6000;
        setError(`Server error ${res.status}`);
        return;
      }

      const json: SpotifyCurrentResponse = await res.json();
      setData(json);
      setError(null);
      nextDelay = json.is_playing ? 1000 : 3000;
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      nextDelay = 6000;
      setError('Connection error — retrying');
    } finally {
      inflightRef.current = false;
      if (!controller.signal.aborted && enabledRef.current) {
        timerRef.current = setTimeout(doPoll, nextDelay);
      }
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      timerRef.current && clearTimeout(timerRef.current);
      abortRef.current?.abort();
      setData(null);
      setError(null);
      return;
    }

    setIsLoading(true);
    doPoll().finally(() => setIsLoading(false));

    return () => {
      timerRef.current && clearTimeout(timerRef.current);
      abortRef.current?.abort();
    };
  }, [enabled, doPoll]);

  return { data, error, isLoading };
}
