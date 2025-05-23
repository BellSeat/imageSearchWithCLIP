// src/app/page.tsx
'use client'; // This component needs client-side functionality

import React, { useState, useEffect } from 'react';
import { useTranslation } from '../hooks/useTranslation'; // Import the custom hook
import { LanguageSwitcher } from '../components/LanguageSwitcher'; // Import the LanguageSwitcher component

// Configuration for your FastAPI backend
const FASTAPI_BASE_URL = 'http://localhost:8000'; // Make sure this matches your FastAPI server address

export default function HomePage() {
  const { t, isLoadingTranslations } = useTranslation(); // Use the custom hook

  const [addFile, setAddFile] = useState<File | null>(null);
  const [addText, setAddText] = useState<string>('');
  const [searchText, setSearchText] = useState<string>('');
  const [searchFile, setSearchFile] = useState<File | null>(null);
  const [searchResults, setSearchResults] = useState<Array<{ path: string; distance: number }>>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>('');
  const [backendStatus, setBackendStatus] = useState<string>('Checking...');

  // Function to clear messages after a few seconds
  const clearMessage = () => {
    setTimeout(() => setMessage(''), 5000);
  };

  // Health check on component mount
  useEffect(() => {
    if (isLoadingTranslations) return; // Don't check health until translations are loaded

    const checkHealth = async () => {
      try {
        const response = await fetch(`${FASTAPI_BASE_URL}/health`);
        if (response.ok) {
          const data = await response.json();
          setBackendStatus(`${t('statusReady')} (Entries: ${data.database_entry_count})`);
        } else {
          setBackendStatus(t('statusDegraded'));
        }
      } catch (error) {
        setBackendStatus(t('statusOffline'));
        console.error('Health check failed:', error);
      }
    };
    checkHealth();
    // Optional: Set up an interval for periodic health checks
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [isLoadingTranslations, t]); // Re-run health check when translations are loaded or language changes

  // Handler for adding a single image
  const handleAddImage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!addFile) {
      setMessage(t('messageSelectImageToAdd'));
      clearMessage();
      return;
    }

    setLoading(true);
    setMessage(t('messageAddingImage'));
    const formData = new FormData();
    formData.append('image', addFile);
    formData.append('text', addText);

    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/add-image`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setMessage(`${t('messageImageAddedSuccess')} ${data.path}`);
        setAddFile(null);
        setAddText('');
      } else {
        const errorData = await response.json();
        setMessage(`${t('messageFailedToAddImage')} ${errorData.detail || response.statusText}`);
      }
    } catch (error: any) {
      console.error('Error adding image:', error);
      setMessage(`${t('messageNetworkError')} ${error.message}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };

  // Handler for searching by text
  const handleSearchText = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchText.trim()) {
      setMessage(t('messageEnterTextForSearch'));
      clearMessage();
      return;
    }

    setLoading(true);
    setMessage(t('messageSearchingByText'));
    setSearchResults([]);

    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/search-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query_text: searchText }),
      });

      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results || []);
        setMessage(data.message);
      } else {
        const errorData = await response.json();
        setMessage(`${t('messageTextSearchFailed')} ${errorData.detail || response.statusText}`);
      }
    } catch (error: any) {
      console.error('Error searching by text:', error);
      setMessage(`${t('messageNetworkError')} ${error.message}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };

  // Handler for searching by image
  const handleSearchImage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchFile) {
      setMessage(t('messageSelectImageForSearch'));
      clearMessage();
      return;
    }

    setLoading(true);
    setMessage(t('messageSearchingByImage'));
    setSearchResults([]);
    const formData = new FormData();
    formData.append('image', searchFile);

    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/search-image`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results || []);
        setMessage(data.message);
      } else {
        const errorData = await response.json();
        setMessage(`${t('messageImageSearchFailed')} ${errorData.detail || response.statusText}`);
      }
    } catch (error: any) {
      console.error('Error searching by image:', error);
      setMessage(`${t('messageNetworkError')} ${error.message}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };

  if (isLoadingTranslations) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <p className="text-xl text-gray-700">Loading translations...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 p-4 font-sans antialiased text-gray-800">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-indigo-700 mb-2">{t('appTitle')}</h1>
        <p className="text-lg text-gray-600">{t('appDescription')}</p>
        <div className="mt-4 text-sm font-medium">
          {t('backendStatus')}: <span className={`font-semibold ${backendStatus.includes(t('statusReady')) ? 'text-green-600' : backendStatus.includes(t('statusDegraded')) ? 'text-yellow-600' : 'text-red-600'}`}>{backendStatus}</span>
        </div>
        <LanguageSwitcher /> {/* Use the dedicated LanguageSwitcher component */}
      </header>

      <main className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-xl">
        {message && (
          <div className="mb-6 p-3 rounded-md bg-blue-100 text-blue-700 border border-blue-200">
            {message}
          </div>
        )}

        <section className="mb-8 p-6 border border-gray-200 rounded-lg">
          <h2 className="text-2xl font-semibold text-indigo-600 mb-4">{t('addSectionTitle')}</h2>
          <form onSubmit={handleAddImage} className="space-y-4">
            <div>
              <label htmlFor="addImageFile" className="block text-sm font-medium text-gray-700 mb-1">{t('selectImageFile')}</label>
              <input
                type="file"
                id="addImageFile"
                accept="image/*"
                onChange={(e) => setAddFile(e.target.files ? e.target.files[0] : null)}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
              />
            </div>
            <div>
              <label htmlFor="addImageText" className="block text-sm font-medium text-gray-700 mb-1">{t('imageDescription')}</label>
              <input
                type="text"
                id="addImageText"
                value={addText}
                onChange={(e) => setAddText(e.target.value)}
                placeholder={t('imageDescriptionPlaceholder')}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? t('processing') : t('addImageButton')}
            </button>
          </form>
        </section>

        <section className="mb-8 p-6 border border-gray-200 rounded-lg">
          <h2 className="text-2xl font-semibold text-indigo-600 mb-4">{t('searchSectionTitle')}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Search by Text */}
            <div>
              <h3 className="text-xl font-medium text-gray-700 mb-3">{t('searchTextSearch')}</h3>
              <form onSubmit={handleSearchText} className="space-y-4">
                <div>
                  <label htmlFor="searchText" className="block text-sm font-medium text-gray-700 mb-1">{t('searchTextSearch')}:</label>
                  <input
                    type="text"
                    id="searchText"
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    placeholder={t('searchTextPlaceholder')}
                    className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? t('processing') : t('searchTextButton')}
                </button>
              </form>
            </div>

            {/* Search by Image */}
            <div>
              <h3 className="text-xl font-medium text-gray-700 mb-3">{t('searchImageSearch')}</h3>
              <form onSubmit={handleSearchImage} className="space-y-4">
                <div>
                  <label htmlFor="searchImageFile" className="block text-sm font-medium text-gray-700 mb-1">{t('selectSearchImage')}</label>
                  <input
                    type="file"
                    id="searchImageFile"
                    accept="image/*"
                    onChange={(e) => setSearchFile(e.target.files ? e.target.files[0] : null)}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100"
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? t('processing') : t('searchImageButton')}
                </button>
              </form>
            </div>
          </div>
        </section>

        {searchResults.length > 0 && (
          <section className="p-6 border border-gray-200 rounded-lg">
            <h2 className="text-2xl font-semibold text-indigo-600 mb-4">{t('resultsSectionTitle')}</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {searchResults.map((result, index) => (
                <div key={index} className="bg-white rounded-lg shadow-md overflow-hidden">
                  <img
                    src={result.path} // This assumes result.path is a direct URL to the image
                    alt={`Search Result ${index + 1}`}
                    className="w-full h-48 object-cover"
                    onError={(e) => {
                      e.currentTarget.onerror = null; // Prevent infinite loop
                      e.currentTarget.src = `https://placehold.co/400x300/cccccc/333333?text=${t('imageNotFound')}`; // Placeholder image
                    }}
                  />
                  <div className="p-4">
                    <p className="text-sm font-medium text-gray-900 truncate" title={result.path}>
                      {t('path')}: {result.path.split('/').pop()?.split('\\').pop()} {/* Display only filename */}
                    </p>
                    <p className="text-sm text-gray-600">{t('distance')}: {result.distance.toFixed(4)}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {searchResults.length === 0 && loading === false && message.includes(t('noResultsFound')) && (
            <section className="p-6 border border-gray-200 rounded-lg text-center text-gray-600">
                <p>{t('noResultsFound')}</p>
            </section>
        )}
      </main>
    </div>
  );
}
