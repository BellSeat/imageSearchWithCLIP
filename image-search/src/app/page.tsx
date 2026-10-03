// src/app/page.tsx
'use client';

import React, { ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { LanguageSwitcher } from '../components/LanguageSwitcher';
import { useTranslation } from '../hooks/useTranslation';

const FASTAPI_BASE_URL = 'http://localhost:8000';

type ImageResult = {
  path: string;
  distance: number;
};

type VideoResult = {
  video_url: string;
  frame_url: string;
  frame_timestamp_s: number;
  clip_url: string | null;
  distance: number;
};

type DirectoryInputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  webkitdirectory?: string;
  directory?: string;
};

type BackendHealthResponse = {
  api_status: 'running' | 'degraded';
  embedding_model_loaded: boolean;
  vector_database_loaded: boolean;
  database_entry_count: number;
  compute_device: string;
  cuda_available: boolean;
  video_worker_count: number;
  details?: string;
};

type BackendHealth = {
  state: 'checking' | 'ready' | 'degraded' | 'offline';
  entryCount?: number;
  device?: string;
  cudaAvailable?: boolean;
  workerCount?: number;
};

type VideoJob = {
  id: string;
  filename: string;
  status: 'queued' | 'running' | 'paused' | 'cancelling' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  stage: string;
  frame_count: number;
  processed_frames: number;
  error?: string | null;
  created_at: string;
  updated_at: string;
};

const directoryPickerProps: DirectoryInputProps = {
  webkitdirectory: '',
  directory: '',
};

const supportedVideoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];

const isVideoFile = (file: File) => {
  const fileName = file.name.toLowerCase();
  return file.type.startsWith('video/')
    || supportedVideoExtensions.some((extension) => fileName.endsWith(extension));
};

const getErrorMessage = (error: unknown) => {
  return error instanceof Error ? error.message : String(error);
};

export default function HomePage() {
  const { t, isLoadingTranslations } = useTranslation();

  const [addFiles, setAddFiles] = useState<FileList | null>(null);
  const [addText, setAddText] = useState('');
  const [videoFiles, setVideoFiles] = useState<FileList | null>(null);
  const [videoProcessingStatus, setVideoProcessingStatus] = useState('');
  const [searchText, setSearchText] = useState('');
  const [searchFile, setSearchFile] = useState<File | null>(null);
  const [searchVideoText, setSearchVideoText] = useState('');
  const [searchVideoImage, setSearchVideoImage] = useState<File | null>(null);
  const [searchResults, setSearchResults] = useState<ImageResult[]>([]);
  const [searchVideoResults, setSearchVideoResults] = useState<VideoResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [backendHealth, setBackendHealth] = useState<BackendHealth>({ state: 'checking' });
  const [uploadProgress, setUploadProgress] = useState('');
  const [videoJobs, setVideoJobs] = useState<VideoJob[]>([]);
  const [videoQueueSubmitting, setVideoQueueSubmitting] = useState(false);

  const selectedVideoFiles = useMemo(
    () => (videoFiles ? Array.from(videoFiles).filter(isVideoFile) : []),
    [videoFiles],
  );

  const hasActiveVideoJobs = videoJobs.some((job) =>
    ['queued', 'running', 'paused', 'cancelling'].includes(job.status),
  );

  const activeVideoJobCount = videoJobs.filter((job) =>
    ['queued', 'running', 'paused', 'cancelling'].includes(job.status),
  ).length;

  const backendStatus = backendHealth.state === 'ready'
    ? `${t('statusReady')} (${backendHealth.entryCount ?? 0} ${t('entries')}) · ${backendHealth.device ?? 'cpu'}`
    : backendHealth.state === 'degraded'
      ? t('statusDegraded')
      : backendHealth.state === 'offline'
        ? t('statusOffline')
        : t('statusChecking');

  const backendStatusClasses = backendHealth.state === 'ready'
    ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
    : backendHealth.state === 'offline'
      ? 'border-rose-200 bg-rose-50 text-rose-700'
      : 'border-amber-200 bg-amber-50 text-amber-700';

  const backendIndicatorClass = backendHealth.state === 'ready'
    ? 'bg-emerald-500'
    : backendHealth.state === 'offline'
      ? 'bg-rose-500'
      : 'bg-amber-500';

  const clearMessage = () => {
    setTimeout(() => setMessage(''), 5000);
  };

  const handleResetDatabase = async (deleteMedia: boolean) => {
    const confirmationMessage = deleteMedia
      ? t('fullResetConfirmation')
      : t('clearDatabaseConfirmation');
    if (!window.confirm(confirmationMessage)) return;

    setLoading(true);
    setMessage(t('resettingDatabase'));

    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/reset-database`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delete_media: deleteMedia }),
      });
      const data = await response.json() as { message?: string; detail?: string };

      if (!response.ok) {
        throw new Error(data.detail || response.statusText);
      }

      setBackendHealth((current) => ({ ...current, state: 'ready', entryCount: 0 }));
      setSearchResults([]);
      setSearchVideoResults([]);
      setMessage(data.message || t('databaseResetComplete'));
    } catch (error: unknown) {
      setMessage(`${t('databaseResetFailed')}: ${getErrorMessage(error)}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };

  const getVideoJobStatusLabel = (status: VideoJob['status']) => {
    const labels: Record<VideoJob['status'], string> = {
      queued: t('jobQueued'),
      running: t('jobRunning'),
      paused: t('jobPaused'),
      cancelling: t('jobCancelling'),
      completed: t('jobCompleted'),
      failed: t('jobFailed'),
      cancelled: t('jobCancelled'),
    };
    return labels[status];
  };

  const refreshVideoJobs = useCallback(async () => {
    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/video-jobs`);
      if (response.ok) setVideoJobs(await response.json() as VideoJob[]);
    } catch (error) {
      console.error('Video job status refresh failed:', error);
    }
  }, []);

  useEffect(() => {
    if (isLoadingTranslations) return;

    let isCurrent = true;
    const checkHealth = async () => {
      try {
        const response = await fetch(`${FASTAPI_BASE_URL}/health`);
        const data: BackendHealthResponse = await response.json();

        if (!isCurrent) return;

        if (response.ok) {
          const isReady = data.api_status === 'running'
            && data.embedding_model_loaded
            && data.vector_database_loaded;
          setBackendHealth(isReady
            ? {
              state: 'ready',
              entryCount: data.database_entry_count,
              device: data.compute_device,
              cudaAvailable: data.cuda_available,
              workerCount: data.video_worker_count,
            }
            : { state: 'degraded' });
        } else {
          setBackendHealth({ state: 'degraded' });
        }
      } catch (error) {
        if (isCurrent) setBackendHealth({ state: 'offline' });
        console.error('Health check failed:', error);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => {
      isCurrent = false;
      clearInterval(interval);
    };
  }, [isLoadingTranslations]);

  useEffect(() => {
    if (isLoadingTranslations) return;
    refreshVideoJobs();
    const interval = setInterval(refreshVideoJobs, hasActiveVideoJobs ? 1000 : 5000);
    return () => clearInterval(interval);
  }, [isLoadingTranslations, hasActiveVideoJobs, refreshVideoJobs]);

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

    for (let i = 0; i < addFiles.length; i += 1) {
      const file = addFiles[i];
      setUploadProgress(`${t('processing')} ${i + 1} / ${addFiles.length}: ${file.name}`);
      setMessage(`${t('addingImage')} ${file.name}`);

      const formData = new FormData();
      formData.append('image', file);
      formData.append('text', addText || `${file.name} (batch upload)`);

      try {
        const response = await fetch(`${FASTAPI_BASE_URL}/add-image`, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          successfulUploads += 1;
        } else {
          failedUploads += 1;
          console.error(`Failed to add ${file.name}:`, await response.json());
        }
      } catch (error) {
        failedUploads += 1;
        console.error(`Network error adding ${file.name}:`, error);
      }
    }

    setLoading(false);
    setAddFiles(null);
    setAddText('');
    setUploadProgress('');
    setMessage(`${t('batchUploadComplete')} ${successfulUploads} ${t('successfullyAdded')}, ${failedUploads} ${t('failedToUpload')}.`);
    clearMessage();
  };

  const handleVideoUploadAndProcess = async (e: FormEvent) => {
    e.preventDefault();
    if (selectedVideoFiles.length === 0) {
      setMessage(t('messageSelectVideoToUpload'));
      clearMessage();
      return;
    }

    setVideoQueueSubmitting(true);
    setMessage(t('queueingVideos'));

    const formData = new FormData();
    selectedVideoFiles.forEach((videoFile) => {
      formData.append('video_files', videoFile);
    });

    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/video-jobs`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json() as {
        message?: string;
        jobs?: VideoJob[];
        detail?: string;
      };

      if (!response.ok) {
        throw new Error(data.detail || response.statusText);
      }

      setVideoJobs((current) => [...(data.jobs || []), ...current]);
      setMessage(data.message || t('videosQueued'));
    } catch (error: unknown) {
      setMessage(`${t('videoQueueFailed')}: ${getErrorMessage(error)}`);
    } finally {
      setVideoQueueSubmitting(false);
    }

    setVideoFiles(null);
    setVideoProcessingStatus('');
  };

  const controlVideoJob = async (jobId: string, action: 'pause' | 'resume' | 'cancel') => {
    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/video-jobs/${jobId}/${action}`, {
        method: 'POST',
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || response.statusText);
      }
      const updatedJob = await response.json() as VideoJob;
      setVideoJobs((current) => current.map((job) => job.id === updatedJob.id ? updatedJob : job));
    } catch (error: unknown) {
      setMessage(`${t('videoJobControlFailed')}: ${getErrorMessage(error)}`);
      clearMessage();
    }
  };

  const handleSearchVideo = async (e: FormEvent) => {
    e.preventDefault();
    if (!searchVideoText.trim() && !searchVideoImage) {
      setMessage(t('messageProvideQueryForVideoSearch'));
      clearMessage();
      return;
    }

    setLoading(true);
    setMessage(t('messageSearchingVideo'));
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
        setMessage(`${t('messageVideoSearchFailed')}: ${errorData.detail || response.statusText}`);
      }
    } catch (error: unknown) {
      console.error('Network error during video search:', error);
      setMessage(`${t('messageNetworkError')}: ${getErrorMessage(error)}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };

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
    } catch (error: unknown) {
      console.error('Error searching by text:', error);
      setMessage(`${t('messageNetworkError')} ${getErrorMessage(error)}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };

  const handlePasteSearchImage = (e: React.ClipboardEvent<HTMLElement>) => {
    const imageItem = Array.from(e.clipboardData.items).find((item) => item.type.startsWith('image/'));
    const pastedImage = imageItem?.getAsFile();

    if (!pastedImage) return;

    e.preventDefault();
    const extension = pastedImage.type.split('/')[1] === 'jpeg' ? 'jpg' : pastedImage.type.split('/')[1] || 'png';
    const pastedFile = new File(
      [pastedImage],
      pastedImage.name || `pasted-image-${Date.now()}.${extension}`,
      { type: pastedImage.type, lastModified: Date.now() },
    );
    setSearchFile(pastedFile);
  };

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
    } catch (error: unknown) {
      console.error('Error searching by image:', error);
      setMessage(`${t('messageNetworkError')}: ${getErrorMessage(error)}`);
    } finally {
      setLoading(false);
      clearMessage();
    }
  };

  if (isLoadingTranslations) {
    return (
      <div className="app-shell flex items-center justify-center">
        <p className="text-lg font-semibold text-slate-700">Loading translations...</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="page-frame">
        <header className="hero-panel">
          <div className="hero-layout">
            <div className="hero-copy">
              <div className="brand-line">
                <span className="brand-mark" aria-hidden="true">C</span>
                <p className="text-sm font-bold uppercase tracking-[0.18em] text-cyan-300">CLIP + FAISS</p>
              </div>
              <h1 className="hero-title">{t('appTitle')}</h1>
              <p className="hero-description">{t('appDescription')}</p>
            </div>
            <div className="hero-actions">
              <div className={`status-pill ${backendStatusClasses}`}>
                <span className={`h-2.5 w-2.5 rounded-full ${backendIndicatorClass}`} />
                {t('backendStatus')}: {backendStatus}
              </div>
              <LanguageSwitcher />
            </div>
          </div>
        </header>

        <main className="space-y-6">
          {(message || uploadProgress || videoProcessingStatus) && (
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
              {message && <div className="notice border-blue-200 bg-blue-50 text-blue-800">{message}</div>}
              {uploadProgress && <div className="notice border-amber-200 bg-amber-50 text-amber-800">{uploadProgress}</div>}
              {videoProcessingStatus && <div className="notice border-teal-200 bg-teal-50 text-teal-800">{videoProcessingStatus}</div>}
            </div>
          )}

          <div className="content-grid">
            <section className="panel feature-panel image-panel">
              <div className="panel-heading">
                <h2 className="panel-title panel-title-large">{t('addSectionTitle')}</h2>
                <span className="rounded-full bg-blue-50 px-3 py-1 text-xs font-bold uppercase tracking-wide text-blue-700">{t('imageTag')}</span>
              </div>
              <p className="panel-subtitle">{t('imageUploadHint')}</p>
              <form onSubmit={handleAddImages} className="mt-5 space-y-4">
                <div>
                  <label htmlFor="addImageFile" className="field-label">{t('selectImageFile')}</label>
                  <input
                    type="file"
                    id="addImageFile"
                    accept="image/*"
                    multiple
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setAddFiles(e.target.files)}
                    className="file-input"
                  />
                </div>
                <div>
                  <label htmlFor="addImageText" className="field-label">{t('imageDescription')}</label>
                  <input
                    type="text"
                    id="addImageText"
                    value={addText}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setAddText(e.target.value)}
                    placeholder={t('imageDescriptionPlaceholder')}
                    className="text-input"
                  />
                </div>
                <button type="submit" disabled={loading || !addFiles || addFiles.length === 0} className="primary-button">
                  {loading ? t('processing') : t('addImageButton')}
                </button>
              </form>
            </section>

            <section className="panel feature-panel video-panel">
              <div className="panel-heading">
                <h2 className="panel-title panel-title-large">{t('uploadVideoSectionTitle')}</h2>
                <span className="rounded-full bg-teal-50 px-3 py-1 text-xs font-bold uppercase tracking-wide text-teal-700">{t('videoTag')}</span>
              </div>
              <p className="panel-subtitle">{t('folderUploadHint')}</p>
              <form onSubmit={handleVideoUploadAndProcess} className="mt-5 space-y-4">
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div>
                    <label htmlFor="uploadVideoFiles" className="field-label">{t('selectVideoFiles')}</label>
                    <input
                      type="file"
                      id="uploadVideoFiles"
                      accept="video/*"
                      multiple
                      onChange={(e: ChangeEvent<HTMLInputElement>) => setVideoFiles(e.target.files)}
                      className="file-input"
                    />
                  </div>
                  <div>
                    <label htmlFor="uploadVideoFolder" className="field-label">{t('selectVideoFolder')}</label>
                    <input
                      type="file"
                      id="uploadVideoFolder"
                      accept="video/*"
                      multiple
                      {...directoryPickerProps}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => setVideoFiles(e.target.files)}
                      className="file-input"
                    />
                  </div>
                </div>
                <div>
                  <p className="mt-2 text-sm text-slate-600">
                    {selectedVideoFiles.length > 0 ? `${t('selectedVideos')}: ${selectedVideoFiles.length}` : t('noVideoSelected')}
                  </p>
                  {selectedVideoFiles.length > 0 && (
                    <div className="selection-list mt-3">
                      <ul className="space-y-1" aria-label={t('selectedVideos')}>
                        {selectedVideoFiles.map((file, index) => {
                          const displayName = file.webkitRelativePath || file.name;
                          return (
                            <li
                              key={`${displayName}-${index}`}
                              className="truncate rounded px-2 py-1.5 text-sm text-slate-700 hover:bg-white"
                              title={displayName}
                            >
                              {index + 1}. {displayName}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </div>
                <button type="submit" disabled={videoQueueSubmitting || selectedVideoFiles.length === 0} className="teal-button">
                  {videoQueueSubmitting ? t('queueingVideos') : t('uploadVideosButton')}
                </button>
              </form>
              {videoJobs.length > 0 && (
                <div className="job-queue">
                  <div className="job-queue-header">
                    <div>
                      <h3 className="text-base font-bold text-slate-900">{t('videoQueueTitle')}</h3>
                      <p className="mt-1 text-xs text-slate-500">
                        {activeVideoJobCount} {t('activeJobs')} · {backendHealth.workerCount ?? 10} {t('workerSlots')}
                      </p>
                    </div>
                    <button type="button" className="text-xs font-bold text-slate-500 hover:text-slate-900" onClick={() => setVideoJobs([])}>
                      {t('clearJobList')}
                    </button>
                  </div>
                  <div className="job-list">
                    {videoJobs.map((job) => (
                      <article key={job.id} className="job-row">
                        <div className="flex min-w-0 items-start justify-between gap-3">
                          <div className="min-w-0">
                            <p className="truncate text-sm font-bold text-slate-900" title={job.filename}>{job.filename}</p>
                            <p className="mt-1 truncate text-xs text-slate-500" title={job.error || job.stage}>{job.error || job.stage}</p>
                          </div>
                          <span className={`job-status job-status-${job.status}`}>{getVideoJobStatusLabel(job.status)}</span>
                        </div>
                        <div className="mt-3 flex items-center gap-3">
                          <div className="job-progress-track">
                            <div className="job-progress-fill" style={{ width: `${job.progress}%` }} />
                          </div>
                          <span className="w-10 text-right text-xs font-bold text-slate-600">{job.progress}%</span>
                        </div>
                        <div className="mt-2 flex items-center justify-between gap-3">
                          <span className="text-xs text-slate-500">
                            {job.processed_frames}/{job.frame_count || '?'} {t('frames')}
                          </span>
                          <div className="flex gap-2">
                            {['queued', 'running'].includes(job.status) && (
                              <button type="button" className="job-control-button" onClick={() => controlVideoJob(job.id, 'pause')}>
                                {t('pauseJob')}
                              </button>
                            )}
                            {job.status === 'paused' && (
                              <button type="button" className="job-control-button" onClick={() => controlVideoJob(job.id, 'resume')}>
                                {t('resumeJob')}
                              </button>
                            )}
                            {['queued', 'running', 'paused', 'cancelling'].includes(job.status) && (
                              <button type="button" className="job-control-button job-control-danger" onClick={() => controlVideoJob(job.id, 'cancel')}>
                                {t('cancelJob')}
                              </button>
                            )}
                          </div>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              )}
            </section>
          </div>

          <section className="panel maintenance-panel">
            <h2 className="panel-title">{t('databaseMaintenanceTitle')}</h2>
            <p className="panel-subtitle">{t('databaseMaintenanceDescription')}</p>
            <div className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-2">
              <button
                type="button"
                disabled={loading}
                onClick={() => handleResetDatabase(false)}
                className="primary-button"
              >
                {t('clearDatabaseButton')}
              </button>
              <button
                type="button"
                disabled={loading}
                onClick={() => handleResetDatabase(true)}
                className="danger-button"
              >
                {t('fullResetButton')}
              </button>
            </div>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2 className="panel-title panel-title-large">{t('searchSectionTitle')}</h2>
                <p className="panel-subtitle">{t('imageSearchPanelHint')}</p>
              </div>
              <span className="hidden rounded-full bg-emerald-50 px-3 py-1 text-xs font-bold uppercase tracking-wide text-emerald-700 sm:inline-flex">{t('imageRetrievalTag')}</span>
            </div>
            <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
              <form onSubmit={handleSearchText} className="tool-column space-y-4">
                <h3 className="text-base font-bold text-slate-800">{t('searchTextSearch')}</h3>
                <div>
                  <label htmlFor="searchText" className="sr-only">{t('searchTextSearch')}</label>
                  <input
                    type="text"
                    id="searchText"
                    value={searchText}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchText(e.target.value)}
                    placeholder={t('searchTextPlaceholder')}
                    className="text-input"
                  />
                </div>
                <button type="submit" disabled={loading} className="success-button">
                  {loading ? t('processing') : t('searchTextButton')}
                </button>
              </form>

              <form onSubmit={handleSearchImage} className="tool-column tool-column-divider space-y-4">
                <h3 className="text-base font-bold text-slate-800">{t('searchImageSearch')}</h3>
                <div
                  tabIndex={0}
                  onPaste={handlePasteSearchImage}
                  className="rounded-md outline-none focus:ring-2 focus:ring-blue-200"
                >
                  <label htmlFor="searchImageFile" className="field-label">{t('selectSearchImage')}</label>
                  <input
                    type="file"
                    id="searchImageFile"
                    accept="image/*"
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchFile(e.target.files ? e.target.files[0] : null)}
                    className="file-input"
                  />
                  <p className="mt-2 text-xs text-slate-500">{t('pasteImageHint')}</p>
                  {searchFile && (
                    <p className="mt-2 truncate text-sm font-medium text-blue-700" title={searchFile.name}>
                      {t('selectedSearchImage')}: {searchFile.name}
                    </p>
                  )}
                </div>
                <button type="submit" disabled={loading} className="success-button">
                  {loading ? t('processing') : t('searchImageButton')}
                </button>
              </form>
            </div>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2 className="panel-title panel-title-large">{t('searchVideoSectionTitle')}</h2>
                <p className="panel-subtitle">{t('videoSearchPanelHint')}</p>
              </div>
              <span className="hidden rounded-full bg-teal-50 px-3 py-1 text-xs font-bold uppercase tracking-wide text-teal-700 sm:inline-flex">{t('videoRetrievalTag')}</span>
            </div>
            <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
              <form onSubmit={handleSearchVideo} className="tool-column space-y-4">
                <h3 className="text-base font-bold text-slate-800">{t('searchVideoByText')}</h3>
                <div>
                  <label htmlFor="searchVideoText" className="sr-only">{t('searchVideoTextLabel')}</label>
                  <input
                    type="text"
                    id="searchVideoText"
                    value={searchVideoText}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => {
                      setSearchVideoText(e.target.value);
                      setSearchVideoImage(null);
                    }}
                    placeholder={t('searchVideoTextPlaceholder')}
                    className="text-input"
                  />
                </div>
                <button type="submit" disabled={loading || !searchVideoText.trim()} className="teal-button">
                  {loading ? t('processing') : t('searchVideoTextButton')}
                </button>
              </form>

              <form onSubmit={handleSearchVideo} className="tool-column tool-column-divider space-y-4">
                <h3 className="text-base font-bold text-slate-800">{t('searchVideoByImage')}</h3>
                <div>
                  <label htmlFor="searchVideoImage" className="field-label">{t('selectSearchVideoImage')}</label>
                  <input
                    type="file"
                    id="searchVideoImage"
                    accept="image/*"
                    onChange={(e: ChangeEvent<HTMLInputElement>) => {
                      setSearchVideoImage(e.target.files ? e.target.files[0] : null);
                      setSearchVideoText('');
                    }}
                    className="file-input"
                  />
                </div>
                <button type="submit" disabled={loading || !searchVideoImage} className="teal-button">
                  {loading ? t('processing') : t('searchVideoImageButton')}
                </button>
              </form>
            </div>
<<<<<<< HEAD
=======
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
                      }
                    }
                    width={440}
                    height={550}
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
>>>>>>> d581699f92f08193a0cc088cefa2adb7bf3eea4a
          </section>

          {searchResults.length > 0 && (
            <section className="panel">
              <h2 className="panel-title">{t('imageSearchResultsTitle')}</h2>
              <div className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
                {searchResults.map((result, index) => (
                  <div key={`${result.path}-${index}`} className="result-card">
                    <img
                      src={result.path}
                      alt={`Search Result ${index + 1}`}
                      className="h-52 w-full object-cover"
                      onError={(e: React.SyntheticEvent<HTMLImageElement>) => {
                        e.currentTarget.onerror = null;
                        e.currentTarget.src = `https://placehold.co/400x300/cccccc/333333?text=${t('imageNotFound')}`;
                      }}
                    />
                    <div className="p-4">
                      <p className="truncate text-sm font-semibold text-slate-900" title={result.path}>
                        {t('path')}: {result.path.split('/').pop()?.split('\\').pop()}
                      </p>
                      <p className="mt-1 text-sm text-slate-600">{t('distance')}: {result.distance.toFixed(4)}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {searchVideoResults.length > 0 && (
            <section className="panel">
              <h2 className="panel-title">{t('videoSearchResultsTitle')}</h2>
              <div className="mt-5 space-y-4">
                {searchVideoResults.map((result, index) => (
                  <div key={`${result.frame_url}-${index}`} className="result-card grid grid-cols-1 md:grid-cols-[20rem_minmax(0,1fr)]">
                    {result.clip_url ? (
                      <video controls className="h-56 w-full bg-black object-contain" src={result.clip_url}>
                        {t('yourBrowserDoesNotSupportVideo')}
                      </video>
                    ) : (
                      <img
                        src={result.frame_url}
                        alt={`Video Frame ${index + 1}`}
                        className="h-56 w-full bg-slate-100 object-contain"
                        onError={(e: React.SyntheticEvent<HTMLImageElement>) => {
                          e.currentTarget.onerror = null;
                          e.currentTarget.src = `https://placehold.co/400x300/cccccc/333333?text=${t('imageNotFound')}`;
                        }}
                      />
                    )}
                    <div className="min-w-0 p-5">
                      <p className="truncate text-sm font-semibold text-slate-900">
                        {t('originalVideo')}:{' '}
                        <a href={result.video_url} target="_blank" rel="noopener noreferrer" className="text-blue-700 hover:underline">
                          {result.video_url.split('/').pop()?.split('\\').pop()}
                        </a>
                      </p>
                      <p className="mt-2 truncate text-sm text-slate-700">
                        {t('matchedFrame')}:{' '}
                        <a href={result.frame_url} target="_blank" rel="noopener noreferrer" className="text-blue-700 hover:underline">
                          {result.frame_url.split('/').pop()?.split('\\').pop()}
                        </a>
                      </p>
                      <p className="mt-2 text-sm text-slate-600">{t('atTime')}: {result.frame_timestamp_s.toFixed(2)}s</p>
                      <p className="mt-1 text-sm text-slate-600">{t('distance')}: {result.distance.toFixed(4)}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {searchResults.length === 0 && searchVideoResults.length === 0 && !loading && message.includes(t('noResultsFound')) && (
            <section className="panel text-center text-slate-600">
              <p>{t('noResultsFound')}</p>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
