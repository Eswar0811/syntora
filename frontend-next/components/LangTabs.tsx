'use client';

import type { Language, LangConfig } from '@/lib/types';

interface Props {
  languages: LangConfig[];
  active: Language;
  detected?: Language;
  onSelect: (lang: Language) => void;
}

export default function LangTabs({ languages, active, detected, onSelect }: Props) {
  return (
    <div
      role="tablist"
      aria-label="Language selection"
      className="card-glass flex gap-1.5 p-1.5"
    >
      {languages.map((lang) => {
        const isActive = lang.key === active;
        const isDetected = lang.key === detected;

        return (
          <button
            key={lang.key}
            role="tab"
            aria-selected={isActive}
            className={`lang-tab flex-1 ${isActive ? lang.tabActive : ''}`}
            onClick={() => onSelect(lang.key)}
            style={isActive ? { borderColor: lang.colorVar, color: lang.colorVar } : {}}
          >
            {/* Native script */}
            <span className="lang-tab-native">{lang.native}</span>
            {/* Latin label + detected dot */}
            <span className="flex items-center gap-1 text-[0.7rem]">
              {lang.label}
              {isDetected && (
                <span
                  className="inline-block h-1.5 w-1.5 rounded-full animate-pulse-dot"
                  style={{ background: lang.colorVar }}
                  title="Detected language"
                />
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}
