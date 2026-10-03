// src/components/LanguageSwitcher.tsx
'use client';

import React from 'react';
import { useTranslation } from '../hooks/useTranslation';

export const LanguageSwitcher: React.FC = () => {
  const { t, language, setLanguage } = useTranslation();

  return (
    <div>
      <label htmlFor="language-select" className="sr-only">{t('language')}</label>
      <select
        id="language-select"
        value={language}
        onChange={(e) => setLanguage(e.target.value)}
        className="rounded-lg border border-white/20 bg-white/10 px-3 py-2 text-sm font-bold text-white shadow-sm outline-none transition focus:border-cyan-300 focus:ring-2 focus:ring-cyan-200/40"
      >
        <option value="en">English</option>
        <option value="zh">中文</option>
      </select>
    </div>
  );
};
