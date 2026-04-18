'use client';

interface Props {
  error: string | null;
}

export default function IdleCard({ error }: Props) {
  return (
    <div className="animate-fade-in card-glass p-8 text-center">
      {/* Animated music bars */}
      <div className="mb-5 flex items-end justify-center gap-1 h-8">
        {[0, 150, 300, 150, 0].map((delay, i) => (
          <span
            key={i}
            className="inline-block w-1 rounded-full bg-[#b3b3b3]/40"
            style={{
              height: `${[40, 70, 100, 70, 40][i]}%`,
              animation: `blink 1.6s ease-in-out ${delay}ms infinite`,
            }}
          />
        ))}
      </div>

      <p className="text-sm font-semibold text-[#b3b3b3]">Nothing playing right now</p>
      <p className="mt-1 text-xs text-[#b3b3b3]/60">
        Start a song on Spotify and it will appear here automatically.
      </p>

      {error && (
        <p className="mt-4 text-xs text-amber-400/80">{error}</p>
      )}
    </div>
  );
}
