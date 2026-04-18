import type { LangConfig, Language } from './types';

export const LANGUAGES: LangConfig[] = [
  {
    key: 'tamil',
    romanKey: 'tanglish',
    label: 'Tamil',
    native: 'தமிழ்',
    colorVar: 'var(--color-tamil)',
    tabActive: 'border-[#c084fc] text-[#c084fc] bg-[rgba(124,58,237,0.15)]',
    displayBg: 'bg-[rgba(124,58,237,0.08)] border border-[rgba(192,132,252,0.2)]',
    glowShadow: '0 0 24px rgba(192,132,252,0.2)',
  },
  {
    key: 'hindi',
    romanKey: 'hinglish',
    label: 'Hindi',
    native: 'हिन्दी',
    colorVar: 'var(--color-hindi)',
    tabActive: 'border-[#38bdf8] text-[#38bdf8] bg-[rgba(2,132,199,0.15)]',
    displayBg: 'bg-[rgba(2,132,199,0.08)] border border-[rgba(56,189,248,0.2)]',
    glowShadow: '0 0 24px rgba(56,189,248,0.2)',
  },
  {
    key: 'malayalam',
    romanKey: 'manglish',
    label: 'Malayalam',
    native: 'മലയാളം',
    colorVar: 'var(--color-malayalam)',
    tabActive: 'border-[#34d399] text-[#34d399] bg-[rgba(5,150,105,0.15)]',
    displayBg: 'bg-[rgba(5,150,105,0.08)] border border-[rgba(52,211,153,0.2)]',
    glowShadow: '0 0 24px rgba(52,211,153,0.2)',
  },
  {
    key: 'telugu',
    romanKey: 'tenglish',
    label: 'Telugu',
    native: 'తెలుగు',
    colorVar: 'var(--color-telugu)',
    tabActive: 'border-[#fb923c] text-[#fb923c] bg-[rgba(234,88,12,0.15)]',
    displayBg: 'bg-[rgba(234,88,12,0.08)] border border-[rgba(251,146,60,0.2)]',
    glowShadow: '0 0 24px rgba(251,146,60,0.2)',
  },
];

export const DETECTED_TO_LANG: Record<string, Language> = {
  ta: 'tamil',
  hi: 'hindi',
  ml: 'malayalam',
  te: 'telugu',
};

export const SOURCE_CONFIG: Record<string, { label: string; cls: string }> = {
  whisper: { label: 'Whisper AI', cls: 'bg-orange-500/20 text-orange-400 border-orange-500/30' },
  curated: { label: 'Curated',   cls: 'bg-amber-500/20 text-amber-400 border-amber-500/30' },
};

