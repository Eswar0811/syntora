export interface SpotifyCurrentResponse {
  is_playing: boolean;
  song?: string;
  artist?: string;
  progress_seconds?: number;
  track_id?: string;
  source?: 'lrclib' | 'whisper' | 'curated';
  detected_language?: string;
  detected_label?: string;
  original?: string;
  tamil?: string;
  tanglish?: string;
  hindi?: string;
  hinglish?: string;
  malayalam?: string;
  manglish?: string;
  telugu?: string;
  tenglish?: string;
  message?: string;
}

export type Language = 'tamil' | 'hindi' | 'malayalam' | 'telugu';

export interface LangConfig {
  key: Language;
  romanKey: keyof SpotifyCurrentResponse;
  label: string;
  native: string;
  colorVar: string;
  tabActive: string;
  displayBg: string;
  glowShadow: string;
}

export type AuthStatus = 'idle' | 'exchanging' | 'authed' | 'error';
