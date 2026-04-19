'use client';

import { useState, useEffect, useRef } from 'react';
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

// How far ahead to look when selecting the displayed line client-side.
// Should match _SYNC_OFFSET on the backend.
const SYNC_OFFSET = 0.9;

export function useSpotifyLive({ enabled, onUnauthorized }: Options): LiveState {
  const poll = useSpotifyPoll({ enabled, onUnauthorized });
  const [display, setDisplay] = useState<SpotifyCurrentResponse | null>(null);

  // Track when the last poll arrived and what position Spotify reported
  const posRef = useRef<{ progress: number; at: number } | null>(null);

  // Update display and position reference whenever a new poll arrives
  useEffect(() => {
    if (!poll.data) { setDisplay(null); return; }

    setDisplay(poll.data);

    if (poll.data.is_playing && poll.data.progress_seconds != null) {
      posRef.current = { progress: poll.data.progress_seconds, at: Date.now() };
    }
  }, [poll.data]);

  // 200ms timer: interpolate playback position and advance to next line early
  useEffect(() => {
    if (!enabled) return;

    const timer = setInterval(() => {
      setDisplay(prev => {
        if (!prev?.is_playing || !posRef.current) return prev;

        // Interpolated position = last known position + elapsed since poll
        const elapsed = (Date.now() - posRef.current.at) / 1000;
        const pos = posRef.current.progress + elapsed + SYNC_OFFSET;

        // Advance to next pre-translated line if we've reached its timestamp
        const nextTime = prev.next_line_time;
        if (
          nextTime != null &&
          prev.next_original &&
          pos >= nextTime &&
          prev.original !== prev.next_original
        ) {
          return {
            ...prev,
            original:  prev.next_original,
            line_time: prev.next_line_time,
            tamil:     prev.next_tamil,
            tanglish:  prev.next_tanglish,
            hindi:     prev.next_hindi,
            hinglish:  prev.next_hinglish,
            malayalam: prev.next_malayalam,
            manglish:  prev.next_manglish,
            telugu:    prev.next_telugu,
            tenglish:  prev.next_tenglish,
            // Clear next so we don't re-trigger
            next_original: undefined,
            next_line_time: undefined,
          };
        }

        // Also keep progress bar smooth
        return { ...prev, progress_seconds: posRef.current!.progress + elapsed };
      });
    }, 200);

    return () => clearInterval(timer);
  }, [enabled]);

  return { data: display, error: poll.error, isLoading: poll.isLoading, transport: 'poll' };
}
