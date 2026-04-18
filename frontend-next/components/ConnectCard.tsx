'use client';

import type { AuthStatus } from '@/lib/types';

interface Props {
  status: AuthStatus;
  error: string | null;
  onConnect: () => void;
}


export default function ConnectCard({ status, error, onConnect }: Props) {
  const isConnecting = status === 'exchanging';

  return (
    <div className="animate-fade-in card-glass p-8 sm:p-12">
      {/* Logo + brand */}
      <div className="mb-6 flex justify-center">
        <img src="/syntora-logo.png" alt="Syntora" className="h-16 w-16 rounded-2xl object-cover" />
      </div>

      <h2 className="mb-2 text-center text-2xl font-bold tracking-tight">
        Your Song, Your Vibe!
      </h2>
      <p className="mb-8 text-center text-sm leading-relaxed text-[#b3b3b3]">
        Connect Spotify and Syntora instantly transliterates your playing song
        into Tamil, Hindi, Malayalam, and Telugu — live.
      </p>

      {/* Language preview pills */}
      <div className="mb-8 flex flex-wrap justify-center gap-2">
        {[
          { label: 'தமிழ்',  color: '#c084fc' },
          { label: 'हिन्दी', color: '#38bdf8' },
          { label: 'മലയാളം', color: '#34d399' },
          { label: 'తెలుగు', color: '#fb923c' },
        ].map(({ label, color }) => (
          <span
            key={label}
            className="rounded-full px-3 py-1 text-sm font-semibold"
            style={{ background: `${color}1a`, color, border: `1px solid ${color}33` }}
          >
            {label}
          </span>
        ))}
      </div>

      {/* Error */}
      {(status === 'error' || error) && (
        <div className="mb-6 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-center">
          <p className="text-sm font-semibold text-red-400">Authentication failed</p>
          {error && <p className="mt-1 text-xs text-red-400/80">{error}</p>}
        </div>
      )}

      {/* CTA */}
      <div className="flex justify-center">
        <button
          className="btn-spotify"
          onClick={onConnect}
          disabled={isConnecting}
          aria-label="Connect with Spotify"
        >
          {isConnecting ? (
            <>
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <circle cx="12" cy="12" r="10" strokeOpacity={0.3} />
                <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
              </svg>
              Connecting…
            </>
          ) : (
            'Connect with Spotify'
          )}
        </button>
      </div>

      <p className="mt-5 text-center text-xs text-[#b3b3b3]/60">
        We only request read-only access to your playback state.
      </p>
    </div>
  );
}
