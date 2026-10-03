// src/app/layout.tsx
import './globals.css';
import { LanguageProvider } from '../context/LanguageContext';

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
      <body suppressHydrationWarning={true}>
        <LanguageProvider>
          {children}
        </LanguageProvider>
      </body>
    </html>
  );
}
