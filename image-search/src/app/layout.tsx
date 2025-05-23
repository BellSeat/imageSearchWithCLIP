// src/app/layout.tsx
// REMOVE 'use client' from here (as previously advised)
import './globals.css';
import { Inter } from 'next/font/google';
import { LanguageProvider } from '../context/LanguageContext';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'Image and Text Search App',
  description: 'Similarity search using CLIP and FAISS',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className} suppressHydrationWarning={true}>
        <LanguageProvider>
          {children}
        </LanguageProvider>
      </body>
    </html>
  );
}