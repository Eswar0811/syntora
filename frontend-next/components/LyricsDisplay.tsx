'use client';

import type { LangConfig, SpotifyCurrentResponse } from '@/lib/types';

interface Props {
  lang: LangConfig;
  nativeText: string | undefined;
  romanText: string | undefined;
  source: SpotifyCurrentResponse['source'];
}

export default function LyricsDisplay({ lang, nativeText, romanText, source }: Props) {
  /* Waiting states */
  if (!nativeText && source === 'whisper') {
    return (
      <EmptyState
        lang={lang}
        message="Transcribing audio…"
        sub="Using Whisper AI to recognise the lyrics"
        loading
      />
    );
  }

  if (!nativeText) {
    return (
      <EmptyState
        lang={lang}
        message="Waiting for lyrics…"
        sub="Searching lrclib for synced lyrics"
        loading
      />
    );
  }

  return (
    <div
      className={`animate-slide-up card-glass p-6 ${lang.displayBg}`}
      style={{ boxShadow: lang.glowShadow }}
    >
      {/* Native script */}
      <p
        className="script-display mb-3 text-white"
        lang={langCode(lang.key)}
        dir="ltr"
      >
        {nativeText}
      </p>

      {/* Divider */}
      <div className="mb-3 h-px bg-white/5" />

      {/* Romanised form */}
      {romanText && (
        <p className="roman-display" style={{ color: '#1db954' }}>
          {romanText}
        </p>
      )}
    </div>
  );
}

/* ── Empty / loading state ── */
function EmptyState({
  lang,
  message,
  sub,
  loading,
}: {
  lang: LangConfig;
  message: string;
  sub: string;
  loading?: boolean;
}) {
  return (
    <div
      className={`animate-fade-in card-glass p-6 text-center ${lang.displayBg}`}
      style={{ boxShadow: lang.glowShadow }}
    >
      {loading && (
        <div className="mb-3 flex justify-center gap-1">
          {[0, 100, 200].map((delay) => (
            <span
              key={delay}
              className="inline-block h-2 w-2 rounded-full bg-[#b3b3b3]/40"
              style={{ animation: `blink 1.4s ease-in-out ${delay}ms infinite` }}
            />
          ))}
        </div>
      )}
      <p className="text-sm font-semibold text-[#b3b3b3]">{message}</p>
      <p className="mt-1 text-xs text-[#b3b3b3]/60">{sub}</p>
    </div>
  );
}

function langCode(key: string): string {
  const map: Record<string, string> = {
    tamil: 'ta', hindi: 'hi', malayalam: 'ml', telugu: 'te',
  };
  return map[key] ?? 'und';
}
