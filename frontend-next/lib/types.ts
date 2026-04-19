export interface SpotifyCurrentResponse {
  is_playing: boolean;
  song?: string;
  artist?: string;
  progress_seconds?: number;
  track_id?: string;
  source?: 'lrclib' | 'whisper' | 'curated';
  detected_language?: string;
  detected_label?: string;
  stale?: boolean;
  sync_offset?: number;

  // Current line
  original?: string;
  line_time?: number;
  tamil?: string;
  tanglish?: string;
  hindi?: string;
  hinglish?: string;
  malayalam?: string;
  manglish?: string;
  telugu?: string;
  tenglish?: string;

  // Next line — pre-translated for instant transition
  next_original?: string;
  next_line_time?: number;
  next_tamil?: string;
  next_tanglish?: string;
  next_hindi?: string;
  next_hinglish?: string;
  next_malayalam?: string;
  next_manglish?: string;
  next_telugu?: string;
  next_tenglish?: string;

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
