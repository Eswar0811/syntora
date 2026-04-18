import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import { Analytics } from '@vercel/analytics/next';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Syntora — Your Song, Your Vibe!',
  description:
    'Real-time Spotify lyrics transliterated into Tamil, Hindi, Malayalam, and Telugu. Connect your Spotify and read along in your native script.',
  keywords: ['Tamil', 'Hindi', 'Malayalam', 'Telugu', 'transliteration', 'Spotify', 'lyrics'],
  authors: [{ name: 'Syntora' }],
  openGraph: {
    title: 'Syntora — Your Song, Your Vibe!',
    description: 'See your Spotify lyrics in Tamil, Hindi, Malayalam, and Telugu — live.',
    type: 'website',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#121212',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-dvh bg-[#121212] text-white antialiased">
        {children}
        <Analytics />
      </body>
    </html>
  );
}
