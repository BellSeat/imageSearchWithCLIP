// src/app/page.tsx
'use client';

import React, { useState, useEffect, ChangeEvent, FormEvent } from 'react';
import { useTranslation } from '../hooks/useTranslation';
import { LanguageSwitcher } from '../components/LanguageSwitcher';

const FASTAPI_BASE_URL = 'http://localhost:8000';

export default function HomePage() {
  const { t, isLoadingTranslations } = useTranslation();

  const [addFiles, setAddFiles] = useState<FileList | null>(null);
  const [addText, setAddText] = useState<string>('');

  const [videoFile, setVideoFile] = useState<File | null>(null); // For video upload
  const [videoProcessingStatus, setVideoProcessingStatus] = useState<string>('');

  const [searchText, setSearchText] = useState<string>('');
  const [searchFile, setSearchFile] = useState<File | null>(null);
  
  const [searchVideoText, setSearchVideoText] = useState<string>(''); // For video search text
  const [searchVideoImage, setSearchVideoImage] = useState<File | null>(null); // For video search image
  const [searchVideoResults, setSearchVideoResults] = useState<Array<{ video_url: string; frame_url: string; frame_timestamp_s: number; clip_url: string | null; distance: number }>>([]);


  const [searchResults, setSearchResults] = useState<Array<{ path: string; distance: number }>>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>('');
  const [backendStatus, setBackendStatus] = useState<string>('Checking...');
  const [uploadProgress, setUploadProgress] = useState<string>('');

  const clearMessage = () => {
    setTimeout(() => setMessage(''), 5000);
  };

  useEffect(() => {
    if (isLoadingTranslations) return;

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
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [isLoadingTranslations, t]);

  // Handler for adding multiple images (now iterates and uploads one by one)
  const handleAddImages = async (e: FormEvent) => {
    e.preventDefault();
    if (!addFiles || addFiles.length === 0) {
      setMessage(t('messageSelectImageToAdd'));
      clearMessage();
      return;
    }

    setLoading(true);
    setUploadProgress('');
    setSearchResults([]);

    let successfulUploads = 0;
    let failedUploads = 0;

    for (let i = 0; i < addFiles.length; i++) {
      const file = addFiles[i];
      setUploadProgress(`${t('processing')} file ${i + 1} of ${addFiles.length}: ${file.name}`);
      setMessage(`${t('addingImage')} ${file.name}...`);

      const formData = new FormData();
      formData.append('image', file);
      formData.append('text', addText || `${file.name} (batch upload)`); 

      try {
        const response = await fetch(`${FASTAPI_BASE_URL}/add-image`, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          console.log(`Successfully added: ${data.uploaded_url}`);
          successfulUploads++;
        } else {
          const errorData = await response.json();
          console.error(`Failed to add ${file.name}:`, errorData); // Log full error for debugging
          failedUploads++;
        }
      } catch (error: any) {
        console.error(`Network error adding ${file.name}:`, error); // Log full error
        failedUploads++;
      }
    }

    setLoading(false);
    setAddFiles(null);
    setAddText('');
    setUploadProgress('');
    setMessage(`${t('batchUploadComplete')} ${successfulUploads} ${t('successfullyAdded')}, ${failedUploads} ${t('failedToUpload')}.`);
    clearMessage();
  };

  // --- New Handler for Video Upload and Processing ---
  const handleVideoUploadAndProcess = async (e: FormEvent) => {
    e.preventDefault();
    if (!videoFile) {
      setMessage(t('messageSelectVideoToUpload')); // You'll need to add this translation
      clearMessage();
      return;
    }

    setLoading(true);
    setVideoProcessingStatus(t('messageUploadingVideo')); // Add this translation
    setUploadProgress(''); // Clear image upload progress

    const formData = new FormData();
    formData.append('video_file', videoFile); // Must match backend parameter name

    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/upload-and-process-video`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setVideoProcessingStatus(`${t('messageVideoProcessedSuccess')} ${data.video_url}. ${t('messageExtractedFrames')}: ${data.frame_count}`); // Add translations
        setMessage(`Video: ${videoFile.name} processed successfully.`);
        setVideoFile(null);
      } else {
        const errorData = await response.json();
        setVideoProcessingStatus(`${t('messageVideoProcessingFailed')}: ${errorData.detail || response.statusText}`); // Add translation
        setMessage(`Failed to process video: ${videoFile.name}`);
        console.error('Video processing failed:', errorData);
      }
    } catch (error: any) {
      console.error('Network error during video upload/processing:', error);
      setVideoProcessingStatus(`${t('messageNetworkError')}: ${error.message}`);
      setMessage(`Network error processing video: ${videoFile.name}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };

  // --- New Handler for Video Search (Text or Image) ---
  const handleSearchVideo = async (e: FormEvent) => {
    e.preventDefault();
    if (!searchVideoText.trim() && !searchVideoImage) {
      setMessage(t('messageProvideQueryForVideoSearch')); // Add translation
      clearMessage();
      return;
    }

    setLoading(true);
    setMessage(t('messageSearchingVideo')); // Add translation
    setSearchVideoResults([]);

    const formData = new FormData();
    if (searchVideoText.trim()) {
      formData.append('query_text', searchVideoText.trim());
    } else if (searchVideoImage) {
      formData.append('query_image', searchVideoImage);
    }

    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/search-video`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setSearchVideoResults(data.results || []);
        setMessage(data.message);
      } else {
        const errorData = await response.json();
        setMessage(`${t('messageVideoSearchFailed')}: ${errorData.detail || response.statusText}`); // Add translation
        console.error('Video search failed:', errorData);
      }
    } catch (error: any) {
      console.error('Network error during video search:', error);
      setMessage(`${t('messageNetworkError')}: ${error.message}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };


  // Handler for searching by text (existing image search)
  const handleSearchText = async (e: FormEvent) => {
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
        headers: { 'Content-Type': 'application/json' },
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

  // Handler for searching by image (existing image search)
  const handleSearchImage = async (e: FormEvent) => {
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
        setMessage(`${t('messageImageSearchFailed')}: ${errorData.detail || response.statusText}`);
      }
    } catch (error: any) {
      console.error('Error searching by image:', error);
      setMessage(`${t('messageNetworkError')}: ${error.message}`);
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
        <LanguageSwitcher />
      </header>

      <main className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-xl">
        {message && (
          <div className="mb-6 p-3 rounded-md bg-blue-100 text-blue-700 border border-blue-200">
            {message}
          </div>
        )}
        {uploadProgress && (
          <div className="mb-6 p-3 rounded-md bg-yellow-100 text-yellow-700 border border-yellow-200">
            {uploadProgress}
          </div>
        )}
        {videoProcessingStatus && (
          <div className="mb-6 p-3 rounded-md bg-purple-100 text-purple-700 border border-purple-200">
            {videoProcessingStatus}
          </div>
        )}

        {/* Section 1: Add Images to Database (Existing, minor changes) */}
        <section className="mb-8 p-6 border border-gray-200 rounded-lg">
          <h2 className="text-2xl font-semibold text-indigo-600 mb-4">{t('addSectionTitle')}</h2>
          <form onSubmit={handleAddImages} className="space-y-4">
            <div>
              <label htmlFor="addImageFile" className="block text-sm font-medium text-gray-700 mb-1">{t('selectImageFile')}:</label>
              <input
                type="file"
                id="addImageFile"
                accept="image/*"
                multiple
                onChange={(e: ChangeEvent<HTMLInputElement>) => setAddFiles(e.target.files)}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
              />
            </div>
            <div>
              <label htmlFor="addImageText" className="block text-sm font-medium text-gray-700 mb-1">{t('imageDescription')}:</label>
              <input
                type="text"
                id="addImageText"
                value={addText}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setAddText(e.target.value)}
                placeholder={t('imageDescriptionPlaceholder')}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !addFiles || addFiles.length === 0}
              className="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? t('processing') : t('addImageButton')}
            </button>
          </form>
        </section>

        {/* New Section 2: Upload and Process Video */}
        <section className="mb-8 p-6 border border-gray-200 rounded-lg">
          <h2 className="text-2xl font-semibold text-purple-600 mb-4">{t('uploadVideoSectionTitle')}</h2> {/* Add this translation */}
          <form onSubmit={handleVideoUploadAndProcess} className="space-y-4">
            <div>
              <label htmlFor="uploadVideoFile" className="block text-sm font-medium text-gray-700 mb-1">{t('selectVideoFile')}:</label> {/* Add this translation */}
              <input
                type="file"
                id="uploadVideoFile"
                accept="video/*"
                onChange={(e: ChangeEvent<HTMLInputElement>) => setVideoFile(e.target.files ? e.target.files[0] : null)}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !videoFile}
              className="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? t('processing') : t('uploadVideoButton')} {/* Add this translation */}
            </button>
          </form>
        </section>

        {/* Section 3: Search Images (Existing) */}
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
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchText(e.target.value)}
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
                  <label htmlFor="searchImageFile" className="block text-sm font-medium text-gray-700 mb-1">{t('selectSearchImage')}:</label>
                  <input
                    type="file"
                    id="searchImageFile"
                    accept="image/*"
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchFile(e.target.files ? e.target.files[0] : null)}
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

        {/* New Section 4: Search Video */}
        <section className="mb-8 p-6 border border-gray-200 rounded-lg">
          <h2 className="text-2xl font-semibold text-teal-600 mb-4">{t('searchVideoSectionTitle')}</h2> {/* Add this translation */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Search Video by Text */}
            <div>
              <h3 className="text-xl font-medium text-gray-700 mb-3">{t('searchVideoByText')}</h3> {/* Add this translation */}
              <form onSubmit={handleSearchVideo} className="space-y-4">
                <div>
                  <label htmlFor="searchVideoText" className="block text-sm font-medium text-gray-700 mb-1">{t('searchVideoTextLabel')}:</label> {/* Add this translation */}
                  <input
                    type="text"
                    id="searchVideoText"
                    value={searchVideoText}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => {
                        setSearchVideoText(e.target.value);
                        setSearchVideoImage(null); // Clear image if text is being typed
                    }}
                    placeholder={t('searchVideoTextPlaceholder')} 
                    className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-teal-500 focus:border-teal-500 sm:text-sm"
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading || !searchVideoText.trim()} // Disable if no text
                  className="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-teal-600 hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? t('processing') : t('searchVideoTextButton')} {/* Add this translation */}
                </button>
              </form>
            </div>

            {/* Search Video by Image */}
            <div>
              <h3 className="text-xl font-medium text-gray-700 mb-3">{t('searchVideoByImage')}</h3> {/* Add this translation */}
              <form onSubmit={handleSearchVideo} className="space-y-4">
                <div>
                  <label htmlFor="searchVideoImage" className="block text-sm font-medium text-gray-700 mb-1">{t('selectSearchVideoImage')}:</label> {/* Add this translation */}
                  <input
                    type="file"
                    id="searchVideoImage"
                    accept="image/*"
                    onChange={(e: ChangeEvent<HTMLInputElement>) => {
                        setSearchVideoImage(e.target.files ? e.target.files[0] : null);
                        setSearchVideoText(''); // Clear text if image is selected
                    }}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-teal-50 file:text-teal-700 hover:file:bg-teal-100"
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading || !searchVideoImage} // Disable if no image
                  className="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-teal-600 hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? t('processing') : t('searchVideoImageButton')} {/* Add this translation */}
                </button>
              </form>
            </div>
          </div>
        </section>

        {/* Section 5: Search Results Display (Split for Image/Video) */}
        {searchResults.length > 0 && (
          <section className="p-6 border border-gray-200 rounded-lg">
            <h2 className="text-2xl font-semibold text-indigo-600 mb-4">{t('imageSearchResultsTitle')}</h2> {/* Add this translation */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {searchResults.map((result, index) => (
                <div key={index} className="bg-white rounded-lg shadow-md overflow-hidden">
                  <img
                    src={result.path}
                    alt={`Search Result ${index + 1}`}
                    className="w-full h-48 object-cover"
                    onError={(e: React.SyntheticEvent<HTMLImageElement, Event>) => {
                      e.currentTarget.onerror = null;
                      e.currentTarget.src = `https://placehold.co/400x300/cccccc/333333?text=${t('imageNotFound')}`;
                    }}
                  />
                  <div className="p-4">
                    <p className="text-sm font-medium text-gray-900 truncate" title={result.path}>
                      {t('path')}: {result.path.split('/').pop()?.split('\\').pop()}
                    </p>
                    <p className="text-sm text-gray-600">{t('distance')}: {result.distance.toFixed(4)}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {searchVideoResults.length > 0 && (
          <section className="p-6 border border-gray-200 rounded-lg mt-8">
            <h2 className="text-2xl font-semibold text-teal-600 mb-4">{t('videoSearchResultsTitle')}</h2> {/* Add this translation */}
            <div className="grid grid-cols-1 sm:grid-cols-1 lg:grid-cols-1 gap-6">
              {searchVideoResults.map((result, index) => (
                <div key={index} className="bg-white rounded-lg shadow-md overflow-hidden flex flex-col md:flex-row">
                  {/* Video Clip Preview */}
                  {result.clip_url ? (
                      <video controls className="w-full md:w-1/2 h-48 object-contain bg-black" src={result.clip_url} onError={(e) => console.error("Video error:", e)}>
                          {t('yourBrowserDoesNotSupportVideo')} {/* Add this translation */}
                      </video>
                  ) : (
                      <img
                          src={result.frame_url} // Fallback to keyframe image if no clip
                          alt={`Video Frame ${index + 1}`}
                          className="w-full md:w-1/2 h-48 object-contain bg-gray-200"
                          onError={(e: React.SyntheticEvent<HTMLImageElement, Event>) => {
                              e.currentTarget.onerror = null;
                              e.currentTarget.src = `https://placehold.co/400x300/cccccc/333333?text=${t('imageNotFound')}`;
                          }}
                      />
                  )}
                  
                  {/* Result Details */}
                  <div className="p-4 flex-1">
                    <p className="text-sm font-medium text-gray-900 mb-1">
                      {t('originalVideo')}: <a href={result.video_url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                          {result.video_url.split('/').pop()?.split('\\').pop()}
                      </a>
                    </p>
                    <p className="text-sm text-gray-700 mb-1">
                      {t('matchedFrame')}: <a href={result.frame_url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                          {result.frame_url.split('/').pop()?.split('\\').pop()}
                      </a> ({t('atTime')}: {result.frame_timestamp_s.toFixed(2)}s)
                    </p>
                    <p className="text-sm text-gray-600">{t('distance')}: {result.distance.toFixed(4)}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}


        {searchResults.length === 0 && searchVideoResults.length === 0 && loading === false && message.includes(t('noResultsFound')) && (
            <section className="p-6 border border-gray-200 rounded-lg text-center text-gray-600">
                <p>{t('noResultsFound')}</p>
            </section>
        )}
      </main>
    </div>
  );
}