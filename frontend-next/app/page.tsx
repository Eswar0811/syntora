import { Suspense } from 'react';
import MainApp from '@/components/MainApp';

export default function Home() {
  return (
    <Suspense fallback={<div className="min-h-dvh bg-[#121212]" />}>
      <MainApp />
    </Suspense>
  );
}
