'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type { SpotifyCurrentResponse } from '../types';
import { sessionHeaders } from '../api';

interface UseSpotifyPollOptions {
  enabled: boolean;
  onUnauthorized: () => void;
}

const MAX_BACKOFF_MS = 30_000;

export function useSpotifyPoll({ enabled, onUnauthorized }: UseSpotifyPollOptions) {
  const [data, setData] = useState<SpotifyCurrentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const enabledRef        = useRef(enabled);
  const inflightRef       = useRef(false);
  const timerRef          = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef          = useRef<AbortController | null>(null);
  const onUnauthorizedRef = useRef(onUnauthorized);
  const errCountRef       = useRef(0);

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
        errCountRef.current += 1;
        nextDelay = Math.min(12_000 * errCountRef.current, MAX_BACKOFF_MS);
        setError('Rate limited — retrying shortly');
        return;
      }

      if (!res.ok) {
        errCountRef.current += 1;
        nextDelay = Math.min(3_000 * Math.pow(2, errCountRef.current - 1), MAX_BACKOFF_MS);
        setError(`Server error ${res.status} — retrying`);
        return;
      }

      const json: SpotifyCurrentResponse = await res.json();
      errCountRef.current = 0;
      setData(json);
      setError(null);
      nextDelay = json.is_playing ? 1000 : 3000;
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      errCountRef.current += 1;
      nextDelay = Math.min(3_000 * Math.pow(2, errCountRef.current - 1), MAX_BACKOFF_MS);
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
      errCountRef.current = 0;
      return;
    }

    setIsLoading(true);
    errCountRef.current = 0;
    doPoll().finally(() => setIsLoading(false));

    return () => {
      timerRef.current && clearTimeout(timerRef.current);
      abortRef.current?.abort();
    };
  }, [enabled, doPoll]);

  return { data, error, isLoading };
}
