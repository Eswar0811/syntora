import { SOURCE_CONFIG } from '@/lib/constants';
import type { SpotifyCurrentResponse } from '@/lib/types';

interface Props {
  source: SpotifyCurrentResponse['source'];
}

export default function SourceBadge({ source }: Props) {
  if (!source) return null;
  const cfg = SOURCE_CONFIG[source];
  if (!cfg) return null;

  return (
    <span className={`source-badge ${cfg.cls}`} title={`Lyrics source: ${cfg.label}`}>
      <svg width="6" height="6" viewBox="0 0 6 6" aria-hidden="true">
        <circle cx="3" cy="3" r="3" fill="currentColor" />
      </svg>
      {cfg.label}
    </span>
  );
}
