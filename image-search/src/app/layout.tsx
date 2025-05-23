// src/app/layout.tsx
import './globals.css'; // Import your global CSS file (e.g., Tailwind CSS)
import { Inter } from 'next/font/google'; // Example font import
import { LanguageProvider } from '../context/LanguageContext'; // Import your LanguageProvider

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
    <html lang="en"> {/* You might want to dynamically set this based on language context if needed */}
      <body className={inter.className}>
        <LanguageProvider>
          {children}
        </LanguageProvider>
      </body>
    </html>
  );
}
