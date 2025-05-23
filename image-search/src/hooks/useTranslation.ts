// src/hooks/useTranslation.ts
'use client'; // This hook is used in client components

import { useState, useEffect, useCallback, useContext } from 'react';
import type { TranslationKeys } from '../locales/en'; // Import the type for key checking
import { LanguageContext } from '../context/LanguageContext'; // Import LanguageContext

// Define the shape of a translation dictionary
interface TranslationDict {
  [key: string]: string;
}

// Map of available languages to their respective translation modules
const languageMap: { [key: string]: () => Promise<{ default: TranslationDict }> } = {
  en: () => import('../locales/en'),
  zh: () => import('../locales/zh'),
  // Add more languages here as needed
};

export const useTranslation = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useTranslation must be used within a LanguageProvider');
  }
  const { language, setLanguage } = context;

  const [translations, setTranslations] = useState<TranslationDict>({});
  const [isLoadingTranslations, setIsLoadingTranslations] = useState<boolean>(true);

  useEffect(() => {
    const loadTranslations = async () => {
      setIsLoadingTranslations(true);
      try {
        const module = languageMap[language];
        if (module) {
          const loadedTranslations = await module();
          setTranslations(loadedTranslations.default);
        } else {
          console.warn(`No translations found for language: ${language}. Falling back to English.`);
          const fallbackModule = languageMap['en'];
          const loadedTranslations = await fallbackModule();
          setTranslations(loadedTranslations.default);
        }
      } catch (error) {
        console.error(`Failed to load translations for ${language}:`, error);
        // Fallback to English on error
        const fallbackModule = languageMap['en'];
        const loadedTranslations = await fallbackModule();
        setTranslations(loadedTranslations.default);
      } finally {
        setIsLoadingTranslations(false);
      }
    };

    loadTranslations();
  }, [language]); // Reload translations when language changes

  // The translation function 't'
  // It takes a key and returns the translated string.
  // It's memoized with useCallback to prevent unnecessary re-renders.
  const t = useCallback((key: TranslationKeys): string => {
    return translations[key] || key; // Return key if translation not found
  }, [translations]);

  return { t, language, setLanguage, isLoadingTranslations };
};
