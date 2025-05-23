// src/components/LanguageSwitcher.tsx
'use client'; // This is a client component

import React from 'react';
import { useTranslation } from '../hooks/useTranslation'; // Import the custom hook

export const LanguageSwitcher: React.FC = () => {
  const { t, language, setLanguage } = useTranslation();

  return (
    <div className="mt-2">
      <label htmlFor="language-select" className="sr-only">{t('language')}</label>
      <select
        id="language-select"
        value={language}
        onChange={(e) => setLanguage(e.target.value)}
        className="block mx-auto mt-2 px-3 py-1 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
      >
        <option value="en">English</option>
        <option value="zh">中文</option>
      </select>
    </div>
  );
};
