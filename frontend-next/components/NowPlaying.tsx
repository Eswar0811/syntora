'use client';

import { useState, useEffect, useRef } from 'react';
import type { SpotifyCurrentResponse, Language } from '@/lib/types';
import { LANGUAGES, DETECTED_TO_LANG } from '@/lib/constants';
import SourceBadge from './SourceBadge';
import LangTabs from './LangTabs';
import LyricsDisplay from './LyricsDisplay';

interface Props {
  data: SpotifyCurrentResponse;
  transport?: 'sse' | 'poll';
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function NowPlaying({ data, transport }: Props) {
  const [activeLang, setActiveLang] = useState<Language>('tamil');
  const fillRef = useRef<HTMLDivElement>(null);

  /* Auto-switch to detected language on song change */
  useEffect(() => {
    if (!data.detected_language) return;
    const mapped = DETECTED_TO_LANG[data.detected_language];
    if (mapped) setActiveLang(mapped);
  }, [data.detected_language, data.track_id]);

  /* Drive progress bar via DOM to avoid inline style prop */
  useEffect(() => {
    if (!fillRef.current || data.progress_seconds == null) return;
    const pct = Math.min(100, (data.progress_seconds / 240) * 100);
    fillRef.current.style.setProperty('--progress-pct', `${pct}%`);
  }, [data.progress_seconds]);

  const lang       = LANGUAGES.find((l) => l.key === activeLang) ?? LANGUAGES[0];
  const nativeText = data[lang.key] as string | undefined;
  const romanText  = data[lang.romanKey] as string | undefined;

  return (
    <div className="animate-fade-in space-y-4">
      {/* ── Track info card ── */}
      <div className="card-glass px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="truncate text-base font-bold leading-tight" title={data.song}>
              {data.song ?? '—'}
            </p>
            <p className="mt-0.5 truncate text-sm text-[#b3b3b3]" title={data.artist}>
              {data.artist ?? '—'}
            </p>
          </div>

          {/* Badges */}
          <div className="flex shrink-0 flex-col items-end gap-1.5">
            {data.source && <SourceBadge source={data.source} />}
            {data.detected_label && (
              <span className="text-[10px] font-semibold uppercase tracking-wider text-[#b3b3b3]/60">
                {data.detected_label}
              </span>
            )}
            {/* Transport indicator — SSE = live dot, poll = clock icon */}
            {transport === 'sse' ? (
              <span className="flex items-center gap-1 text-[10px] font-medium text-[#1db954]/70">
                <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[#1db954]" />
                live
              </span>
            ) : transport === 'poll' ? (
              <span className="text-[10px] text-[#b3b3b3]/40">polling</span>
            ) : null}
          </div>
        </div>

        {/* Progress bar */}
        {data.progress_seconds != null && (
          <div className="mt-3 flex items-center gap-2">
            <span className="w-8 text-right text-[10px] tabular-nums text-[#b3b3b3]/70">
              {formatTime(data.progress_seconds)}
            </span>
            <div className="progress-bar-track flex-1">
              <div ref={fillRef} className="progress-bar-fill" />
            </div>
          </div>
        )}
      </div>

      {/* ── Language tabs ── */}
      <LangTabs
        languages={LANGUAGES}
        active={activeLang}
        onSelect={setActiveLang}
        detected={data.detected_language ? DETECTED_TO_LANG[data.detected_language] : undefined}
      />

      {/* ── Lyrics display ── */}
      <LyricsDisplay
        lang={lang}
        nativeText={nativeText}
        romanText={romanText}
        source={data.source}
      />
    </div>
  );
}
