import os
import sys
import json
import uuid
import shutil
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse # For serving video clips directly


import datetime # For video clip naming

# Workaround for OpenMP runtime error (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory to sys.path for core module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.embedding import Embedding
from core.vector_database import VectorDatabase
import core.preprocessForVideo as preprocess
from core.log_util import setup_local_logger

logger = setup_local_logger(log_path="logs/api_run.log", logger_name="fastapi_app")

# --- Configuration Paths ---
SCRIPT_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(SCRIPT_DIR,"..", "config", "config.json")

UPLOAD_TEMP_DIR = os.path.join(SCRIPT_DIR, "temp_uploads") 

# Root for all static content served by FastAPI
STATIC_CONTENT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "database"))

# Specific static content sub-directories
FRAMES_DIR = os.path.join(STATIC_CONTENT_ROOT, "raw", "video", "frames")
UPLOADED_IMAGES_DIR = os.path.join(STATIC_CONTENT_ROOT, "uploaded_images")

# Ensure all necessary directories exist
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)
app = FastAPI(
    title="Image and Text Search API",
    description="API for embedding images/text and performing similarity searches using CLIP and FAISS.",
    version="1.0.0"
)

# --- New Directories for Video Processing ---
VIDEO_UPLOAD_DIR = os.path.join(STATIC_CONTENT_ROOT, "uploaded_videos") # Where raw uploaded videos go
VIDEO_CLIPS_DIR = os.path.join(STATIC_CONTENT_ROOT, "video_clips")     # Where generated video clips go

os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_CLIPS_DIR, exist_ok=True)

# Add VIDEO_UPLOAD_DIR to the static files if you want to serve raw videos
app.mount("/static/videos", StaticFiles(directory=VIDEO_UPLOAD_DIR), name="static_videos")
logger.info(f"Serving uploaded raw videos from: {VIDEO_UPLOAD_DIR} under /static/videos/")

# Add VIDEO_CLIPS_DIR to the static files for serving clipped segments
app.mount("/static/clips", StaticFiles(directory=VIDEO_CLIPS_DIR), name="static_clips")
logger.info(f"Serving video clips from: {VIDEO_CLIPS_DIR} under /static/clips/")

# --- Global Instances ---
embedder: Optional[Embedding] = None
vectordb: Optional[VectorDatabase] = None



# --- Pydantic Models for Request/Response Schemas ---
class AddImageResponse(BaseModel):
    message: str
    uploaded_url: str
    database_entry_id: Optional[str] = None

class AddFolderRequest(BaseModel):
    folder_path: str

class AddFolderResponse(BaseModel):
    message: str
    processed_count: int
    failed_count: int

class SearchQueryTextRequest(BaseModel):
    query_text: str

class SearchResultItem(BaseModel):
    path: str
    distance: float

class SearchResponse(BaseModel):
    message: str
    results: List[SearchResultItem]
    query_type: str
    query_input: str

class HealthCheckResponse(BaseModel):
    api_status: str
    embedding_model_loaded: bool
    vector_database_loaded: bool
    database_entry_count: int
    details: Optional[str] = None

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES MOUNTS ---
app.mount("/static/frames", StaticFiles(directory=FRAMES_DIR), name="static_frames")
logger.info(f"Serving static frames from: {FRAMES_DIR} under /static/frames/")

app.mount("/static/uploads", StaticFiles(directory=UPLOADED_IMAGES_DIR), name="static_uploads")
logger.info(f"Serving uploaded images from: {UPLOADED_IMAGES_DIR} under /static/uploads/")

# --- Helper Function for File Uploads ---
async def save_upload_file_temporarily(upload_file: UploadFile) -> str:
    """Saves an uploaded file to a temporary location and returns its path."""
    try:
        file_extension = os.path.splitext(upload_file.filename)[1]
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_filepath = os.path.join(UPLOAD_TEMP_DIR, temp_filename)

        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        logger.info(f"Temporarily saved uploaded file to: {temp_filepath}")
        return temp_filepath
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save uploaded file: {e}")

# --- Helper Function to Convert File System Path to Web URL ---
def get_web_url_from_fs_path(fs_path: str) -> str:
    """
    Converts a server-side file system path to a web-accessible absolute URL.
    Assumes the FastAPI app is running on http://localhost:8000.
    """
    base_url = "http://localhost:8000" # Match your FastAPI's actual base URL

    # Normalize paths for consistent comparison, especially on Windows
    normalized_fs_path = os.path.normpath(fs_path)
    normalized_frames_dir = os.path.normpath(FRAMES_DIR)
    normalized_uploaded_images_dir = os.path.normpath(UPLOADED_IMAGES_DIR)

    web_url = fs_path # Default to raw path if no match

    if normalized_fs_path.startswith(normalized_frames_dir):
        relative_path = os.path.relpath(normalized_fs_path, normalized_frames_dir).replace("\\", "/")
        web_url = f"{base_url}/static/frames/{relative_path}"
        logger.debug(f"Converted '{fs_path}' (frames) to URL: {web_url}")
    elif normalized_fs_path.startswith(normalized_uploaded_images_dir):
        relative_path = os.path.relpath(normalized_fs_path, normalized_uploaded_images_dir).replace("\\", "/")
        web_url = f"{base_url}/static/uploads/{relative_path}"
        logger.debug(f"Converted '{fs_path}' (uploads) to URL: {web_url}")
    else:
        logger.warning(f"Could not determine static URL for path: {fs_path}. Returning raw path.")
        
    return web_url

# video model weith Pydantic
class UploadVideoResponse(BaseModel):
    message: str
    video_url: str # URL to the uploaded video
    processed_frames_dir: str # Path to the directory where frames are stored
    frame_count: int
    task_id: str # A task ID for tracking (optional, but good practice)

class SearchVideoResultItem(BaseModel):
    video_url: str        # URL to the original video
    frame_url: str        # URL to the matched keyframe
    frame_timestamp_s: float # Timestamp of the matched frame in seconds
    clip_url: Optional[str] = None # URL to the generated 4-second clip
    distance: float       # Distance from the search query

class SearchVideoResponse(BaseModel):
    message: str
    results: List[SearchVideoResultItem]
    query_type: str
    query_input: str

# ... (startup_event, shutdown_event - no major changes needed here) ...

# --- Helper Function for Video Paths (Update get_web_url_from_fs_path) ---
def get_web_url_from_fs_path(fs_path: str) -> str:
    base_url = "http://localhost:8000"

    normalized_fs_path = os.path.normpath(fs_path)
    normalized_frames_dir = os.path.normpath(FRAMES_DIR)
    normalized_uploaded_images_dir = os.path.normpath(UPLOADED_IMAGES_DIR)
    normalized_video_upload_dir = os.path.normpath(VIDEO_UPLOAD_DIR) # New
    normalized_video_clips_dir = os.path.normpath(VIDEO_CLIPS_DIR)     # New

    web_url = fs_path

    if normalized_fs_path.startswith(normalized_frames_dir):
        relative_path = os.path.relpath(normalized_fs_path, normalized_frames_dir).replace("\\", "/")
        web_url = f"{base_url}/static/frames/{relative_path}"
        logger.debug(f"Converted '{fs_path}' (frames) to URL: {web_url}")
    elif normalized_fs_path.startswith(normalized_uploaded_images_dir):
        relative_path = os.path.relpath(normalized_fs_path, normalized_uploaded_images_dir).replace("\\", "/")
        web_url = f"{base_url}/static/uploads/{relative_path}"
        logger.debug(f"Converted '{fs_path}' (uploads) to URL: {web_url}")
    elif normalized_fs_path.startswith(normalized_video_upload_dir): # New
        relative_path = os.path.relpath(normalized_fs_path, normalized_video_upload_dir).replace("\\", "/")
        web_url = f"{base_url}/static/videos/{relative_path}"
        logger.debug(f"Converted '{fs_path}' (raw video) to URL: {web_url}")
    elif normalized_fs_path.startswith(normalized_video_clips_dir): # New
        relative_path = os.path.relpath(normalized_fs_path, normalized_video_clips_dir).replace("\\", "/")
        web_url = f"{base_url}/static/clips/{relative_path}"
        logger.debug(f"Converted '{fs_path}' (video clip) to URL: {web_url}")
    else:
        logger.warning(f"Could not determine static URL for path: {fs_path}. Returning raw path.")

    return web_url

# --- FastAPI Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup initiated.")
    global embedder, vectordb
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from: {CONFIG_PATH}")

        embedder = await run_in_threadpool(Embedding, config_path=CONFIG_PATH)
        logger.info("CLIP Embedding model initialized successfully.")

        vectordb = await run_in_threadpool(VectorDatabase, config_path=CONFIG_PATH)
        await run_in_threadpool(vectordb.load)
        logger.info(f"Vector Database initialized and loaded. Current entries: {vectordb.get_total_count()}")

        logger.info(f"Configured FRAMES_DIR: {FRAMES_DIR}")
        logger.info(f"Configured UPLOADED_IMAGES_DIR: {UPLOADED_IMAGES_DIR}")

    except FileNotFoundError:
        logger.critical(f"Config file not found at {CONFIG_PATH}. Please ensure it exists.", exc_info=True)
        sys.exit(1)
    except json.JSONDecodeError:
        logger.critical(f"Config file at {CONFIG_PATH} is invalid JSON.", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Failed to initialize core components during startup: {e}", exc_info=True)
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutdown initiated.")
    global vectordb
    if vectordb:
        try:
            await run_in_threadpool(vectordb.save)
            logger.info("Vector Database saved successfully during shutdown.")
        except Exception as e:
            logger.error(f"Failed to save Vector Database during shutdown: {e}", exc_info=True)
    logger.info("FastAPI application shutdown completed.")


# --- API Endpoints ---

@app.post("/add-image", response_model=AddImageResponse, summary="Add a single image to the database")
async def add_single_image(
    image: UploadFile = File(..., description="Image file to add to the database."),
    text: Optional[str] = Form("", description="Optional text description for the image.")
):
    if embedder is None or vectordb is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready. Embedding model or database not initialized.")

    temp_image_path = None
    try:
        temp_image_path = await save_upload_file_temporarily(image)
        
        logger.info(f"Generating embedding for image: {temp_image_path}")
        image_embedding = await run_in_threadpool(embedder.embed_image, temp_image_path)
        
        if image_embedding is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate embedding for the image.")
        
        final_image_filename = f"{uuid.uuid4()}{os.path.splitext(image.filename)[1]}"
        persistent_image_path = os.path.join(UPLOADED_IMAGES_DIR, final_image_filename)
        
        os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)
        shutil.move(temp_image_path, persistent_image_path)
        logger.info(f"Moved uploaded image to persistent storage: {persistent_image_path}")

        metadata = {"path": persistent_image_path, "text": text} 
        
        logger.info(f"Adding image embedding to database. Path: {persistent_image_path}")
        await run_in_threadpool(vectordb.add_vectors, [image_embedding], [metadata])
        await run_in_threadpool(vectordb.save)
        
        web_accessible_url = get_web_url_from_fs_path(persistent_image_path)

        return AddImageResponse(
            message="Image embedded and added to database.",
            uploaded_url=web_accessible_url,
            database_entry_id=final_image_filename
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error adding single image: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.info(f"Removed leftover temporary file: {temp_image_path}")
            except Exception as e:
                logger.error(f"Failed to remove leftover temp file {temp_image_path}: {e}")


@app.post("/add-images-from-folder", response_model=AddFolderResponse, summary="Add all images from a server-side folder")
async def add_images_from_folder(
    request: AddFolderRequest
):
    folder_path = request.folder_path
    if embedder is None or vectordb is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready. Embedding model or database not initialized.")

    if not os.path.isabs(folder_path):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Folder path must be an absolute path.")
    
    if not await run_in_threadpool(os.path.exists, folder_path) or not await run_in_threadpool(os.path.isdir, folder_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Folder not found or is not a directory: {folder_path}")

    batch_embeddings = []
    batch_metadata = []
    processed_count = 0
    failed_count = 0

    try:
        for filename in await run_in_threadpool(os.listdir, folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                image_path = os.path.join(folder_path, filename)
                logger.info(f"Attempting to embed: {image_path}")
                
                image_embedding = await run_in_threadpool(embedder.embed_image, image_path)
                
                if image_embedding is not None:
                    metadata = {"path": image_path, "text": f"Image from {folder_path}"} 
                    batch_embeddings.append(image_embedding)
                    batch_metadata.append(metadata)
                    processed_count += 1
                else:
                    logger.warning(f"Failed to embed image: {filename} in folder {folder_path}")
                    failed_count += 1
        
        if batch_embeddings:
            logger.info(f"Adding {len(batch_embeddings)} embeddings to database in batch...")
            await run_in_threadpool(vectordb.add_vectors, batch_embeddings, batch_metadata)
            await run_in_threadpool(vectordb.save)
            logger.info(f"Successfully processed {processed_count} images from folder and added to database. Failed: {failed_count}")
        else:
            logger.info(f"No images were successfully embedded from folder: {folder_path}")
        
        return AddFolderResponse(
            message="Folder processing completed.",
            processed_count=processed_count,
            failed_count=failed_count
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing images from folder {folder_path}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error while processing folder: {e}")


@app.post("/search-text", response_model=SearchResponse, summary="Search images by text query")
async def search_by_text(
    request: SearchQueryTextRequest
):
    query_text = request.query_text
    if embedder is None or vectordb is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready. Embedding model or database not initialized.")

    logger.info(f"Generating embedding for text query: \"{query_text}\"")
    query_embedding = await run_in_threadpool(embedder.embed_text, query_text)
    
    if query_embedding is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate embedding for the text query.")
    
    logger.info(f"Searching database for text query: \"{query_text}\"")
    search_results = await run_in_threadpool(vectordb.search, query_embedding, top_k=5)
    
    formatted_results = []
    for item in search_results:
        db_path = item[0]["path"]
        distance = item[1]
        web_url = get_web_url_from_fs_path(db_path)
        formatted_results.append(SearchResultItem(path=web_url, distance=distance))
    
    if not formatted_results:
        return SearchResponse(
            message=f"No results found for text query: \"{query_text}\".",
            results=[],
            query_type="text",
            query_input=query_text
        )
    
    return SearchResponse(
        message=f"Search results for text query: \"{query_text}\".",
        results=formatted_results,
        query_type="text",
        query_input=query_text
    )


@app.post("/search-image", response_model=SearchResponse, summary="Search images by image query")
async def search_by_image(
    image: UploadFile = File(..., description="Image file to use as a query for similarity search.")
):
    if embedder is None or vectordb is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready. Embedding model or database not initialized.")

    temp_image_path = None
    try:
        temp_image_path = await save_upload_file_temporarily(image)
        
        logger.info(f"Generating embedding for image query: {temp_image_path}")
        query_embedding = await run_in_threadpool(embedder.embed_image, temp_image_path)
        
        if query_embedding is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate embedding for the image query.")
        
        logger.info(f"Searching database for image query: {image.filename}")
        search_results = await run_in_threadpool(vectordb.search, query_embedding, top_k=5)
        
        formatted_results = []
        for item in search_results:
            db_path = item[0]["path"]
            distance = item[1]
            web_url = get_web_url_from_fs_path(db_path)
            formatted_results.append(SearchResultItem(path=web_url, distance=distance))
        
        if not formatted_results:
            return SearchResponse(
                message=f"No results found for image query: {image.filename}.",
                results=[],
                query_type="image",
                query_input=image.filename
            )
        
        return SearchResponse(
            message=f"Search results for image query: {image.filename}.",
            results=formatted_results,
            query_type="image",
            query_input=image.filename
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error searching by image: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.info(f"Removed temporary query image file: {temp_image_path}")
            except Exception as e:
                logger.error(f"Failed to remove leftover temp query image file {temp_image_path}: {e}")

@app.get("/health", response_model=HealthCheckResponse, summary="Check API and core component health")
async def health_check():
    status_details = "All core components initialized and ready."
    is_ready = True
    entry_count = 0

    if embedder is None or vectordb is None:
        is_ready = False
        status_details = "Embedding model or Vector Database not fully initialized."
    else:
        try:
            entry_count = await run_in_threadpool(vectordb.get_total_count)
        except Exception as e:
            is_ready = False
            status_details = f"Database initialized but failed to get entry count: {e}"
            logger.error(status_details, exc_info=True)

    http_status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthCheckResponse(
        api_status="running" if is_ready else "degraded",
        embedding_model_loaded=embedder is not None,
        vector_database_loaded=vectordb is not None,
        database_entry_count=entry_count,
        details=status_details
    )

@app.post("/upload-and-process-video", response_model=UploadVideoResponse, summary="Upload and process video for embedding")
async def upload_and_process_video(
    video_file: UploadFile = File(..., description="Video file to upload and process.")
):
    """
    Uploads a video file, extracts keyframes, embeds them, and stores in the database.
    Deletes temporary frames after processing.
    """
    if not preprocess.check_ffmpeg():
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="FFmpeg is not installed on the server. Video processing is unavailable.")

    if embedder is None or vectordb is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready. Embedding model or database not initialized.")

    if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported video format. Only MP4, AVI, MOV, MKV are supported.")

    unique_id = str(uuid.uuid4())
    video_filename = f"{unique_id}_{video_file.filename}"
    uploaded_video_path = os.path.join(VIDEO_UPLOAD_DIR, video_filename)

    # 1. Save uploaded video
    try:
        with open(uploaded_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        logger.info(f"Video uploaded to: {uploaded_video_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded video: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save video: {e}")

    processed_frames_dir = None
    try:
        # 2. Extract keyframes
        # preprocess.build_output_dir creates a new folder like database/raw/video/frames/video_name_timestamp_frames_totalframes
        processed_frames_dir_root = os.path.join(FRAMES_DIR, f"{os.path.splitext(video_filename)[0]}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_frames_")
        processed_frames_dir = await run_in_threadpool(preprocess.build_output_dir, uploaded_video_path) # Build output dir needs video_path to get info.
        
        extracted_frame_paths = await run_in_threadpool(preprocess.extract_keyframes, uploaded_video_path, processed_frames_dir)
        
        if not extracted_frame_paths:
            logger.warning(f"No keyframes extracted for video: {uploaded_video_path}")
            # Optionally remove video if no frames found
            os.remove(uploaded_video_path)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No keyframes could be extracted from the video.")

        # 3. Embed keyframes and store in DB
        batch_embeddings = []
        batch_metadata = []
        
        for frame_path in extracted_frame_paths:
            frame_embedding = await run_in_threadpool(embedder.embed_image, frame_path)
            if frame_embedding is not None:
                # Store original video path and frame timestamp in metadata
                frame_basename = os.path.basename(frame_path) # e.g., frame_0001_0-00-00.jpg
                # Extract timestamp from filename: frame_0001_0-00-00.jpg -> 0-00-00.jpg -> 0-00-00 -> seconds
                # This assumes format_time gives seconds with correct parsing
                try:
                    # Parse timestamp from filename (e.g., frame_0001_0-00-00-501500.jpg)
                    # Need to reverse format_time logic or get timestamp direct from ffprobe results
                    # For simplicity, let's pass the frame_path and video_path in metadata
                    # A better way would be to get the timestamp directly from the ffprobe result for that frame
                    # For now, let's assume the timestamp is encoded in the filename via format_time
                    timestamp_str_in_filename = "_".join(frame_basename.split("_")[2:]).replace(".jpg", "")
                    # Converting back to float seconds is complex from format_time, better to pass raw timestamp from ffprobe
                    # Let's adjust metadata to include original video path and the frame file path for later use.
                    # We will rely on getting the timestamp from the frame filename for now, which is a bit brittle.
                    # A more robust solution involves storing a mapping of frame_path to original_timestamp in the DB or a separate index.
                    # For now, we store just the frame path.
                    
                    # Store information needed to reconstruct video/timestamp
                    metadata = {
                        "path": frame_path, # Path to the keyframe image
                        "video_path": uploaded_video_path, # Path to the original video
                        "frame_filename": frame_basename, # Filename of the keyframe
                        "is_keyframe": True,
                        # Add a clean timestamp value if get_keyframe_timestamps returned it directly
                        # "timestamp_s": ...
                    }
                    batch_embeddings.append(frame_embedding)
                    batch_metadata.append(metadata)
                except Exception as e:
                    logger.error(f"Error preparing metadata for frame {frame_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Failed to embed keyframe: {frame_path}. Skipping.")
        
        if batch_embeddings:
            await run_in_threadpool(vectordb.add_vectors, batch_embeddings, batch_metadata)
            await run_in_threadpool(vectordb.save)
            logger.info(f"Successfully embedded and stored {len(batch_embeddings)} keyframes for {video_file.filename}")
        else:
            logger.warning(f"No keyframes were successfully embedded for video: {video_file.filename}")

        # 4. Return success response (and optionally delete video file if it's too large)
        # For this example, we keep the raw video file and frames for static serving and future clipping.
        # If you want to delete the frames, add os.remove(frame_path) loop here.
        # If you want to delete the original video, add os.remove(uploaded_video_path) here.
        
        video_url = get_web_url_from_fs_path(uploaded_video_path)
        return UploadVideoResponse(
            message=f"Video '{video_file.filename}' processed and keyframes embedded.",
            video_url=video_url,
            processed_frames_dir=processed_frames_dir,
            frame_count=len(extracted_frame_paths),
            task_id=unique_id # Using the unique ID for this processing task
        )
    except HTTPException as e:
        if processed_frames_dir and os.path.exists(processed_frames_dir):
            shutil.rmtree(processed_frames_dir) # Clean up generated frames on error
            logger.info(f"Cleaned up frames directory {processed_frames_dir} due to error.")
        if os.path.exists(uploaded_video_path):
            os.remove(uploaded_video_path) # Clean up uploaded video on error
            logger.info(f"Cleaned up uploaded video {uploaded_video_path} due to error.")
        raise e
    except Exception as e:
        logger.error(f"Error processing video {video_file.filename}: {e}", exc_info=True)
        if processed_frames_dir and os.path.exists(processed_frames_dir):
            shutil.rmtree(processed_frames_dir)
            logger.info(f"Cleaned up frames directory {processed_frames_dir} due to error.")
        if os.path.exists(uploaded_video_path):
            os.remove(uploaded_video_path)
            logger.info(f"Cleaned up uploaded video {uploaded_video_path} due to error.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error processing video: {e}")

# --- New API Endpoint: Search Video ---
@app.post("/search-video", response_model=SearchVideoResponse, summary="Search videos by text or image query")
async def search_video(
    query_text: Optional[str] = Body(None, description="Text query for video search."),
    query_image: Optional[UploadFile] = File(None, description="Image file to use as query for video search.")
):
    """
    Searches for relevant video segments (keyframes) based on text or image query.
    Returns details of matched keyframes and generates a 4-second video clip around them.
    """
    if embedder is None or vectordb is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready. Embedding model or database not initialized.")

    if not query_text and not query_image:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either 'query_text' or 'query_image' must be provided.")
    if query_text and query_image:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only one of 'query_text' or 'query_image' can be provided.")

    query_embedding = None
    query_type = "unknown"
    query_input = ""
    temp_query_image_path = None

    try:
        if query_text:
            logger.info(f"Generating embedding for video search text query: \"{query_text}\"")
            query_embedding = await run_in_threadpool(embedder.embed_text, query_text)
            query_type = "text"
            query_input = query_text
        elif query_image:
            temp_query_image_path = await save_upload_file_temporarily(query_image)
            logger.info(f"Generating embedding for video search image query: {temp_query_image_path}")
            query_embedding = await run_in_threadpool(embedder.embed_image, temp_query_image_path)
            query_type = "image"
            query_input = query_image.filename

        if query_embedding is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate embedding for the {query_type} query.")

        logger.info(f"Searching database for video-related query (type: {query_type}, input: {query_input})")
        # Perform search on all keyframes in the database
        search_results_raw = await run_in_threadpool(vectordb.search, query_embedding, top_k=5) # Adjust top_k as needed

        final_results: List[SearchVideoResultItem] = []
        for item in search_results_raw:
            metadata = item[0]  # This is the stored metadata dict (path, video_path, frame_filename, etc.)
            distance = item[1]

            # Ensure this result is actually a keyframe from a video (optional, but good if DB mixes data types)
            if not metadata.get("is_keyframe"):
                logger.debug(f"Skipping non-keyframe result: {metadata.get('path')}")
                continue

            frame_fs_path = metadata.get("path")
            original_video_fs_path = metadata.get("video_path")
            
            if not frame_fs_path or not original_video_fs_path:
                logger.warning(f"Missing path info for search result: {metadata}. Skipping.")
                continue
            
            # --- Get Frame Timestamp (this is the trickiest part) ---
            # Ideally, your metadata should store the exact timestamp (float seconds) of the frame.
            # If not, you might need to re-parse from filename or re-run ffprobe here (inefficient).
            # Assuming frame_filename is like 'frame_0001_0-00-00-501500.jpg'
            frame_basename = os.path.basename(frame_fs_path)
            timestamp_part = "_".join(frame_basename.split("_")[2:]).replace(".jpg", "") # e.g. "0-00-00-501500"
            
            # Convert timestamp string to seconds (rough conversion, improve if format_time changes)
            parts = timestamp_part.split('-')
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                microseconds = int(parts[3]) if len(parts) > 3 else 0
                frame_timestamp_s = float(hours * 3600 + minutes * 60 + seconds + microseconds / 1_000_000)
            except Exception as e:
                logger.warning(f"Could not parse timestamp from filename {frame_basename}: {e}. Skipping clip generation for this frame.")
                frame_timestamp_s = 0.0 # Default to 0.0 if parsing fails, or skip
                # If timestamp parsing fails, we cannot clip reliably. Consider skipping this result or just providing frame URL.
                continue 
            
            # 5. Generate and serve a 4-second video clip
            clip_filename = f"clip_{os.path.basename(original_video_fs_path).replace('.', '_')}_{frame_timestamp_s:.2f}s.mp4"
            clip_fs_path = os.path.join(VIDEO_CLIPS_DIR, clip_filename)
            clip_url = None

            # Check if clip already exists to avoid re-clipping
            if not os.path.exists(clip_fs_path):
                try:
                    await run_in_threadpool(preprocess.clip_video_segment, original_video_fs_path, frame_timestamp_s, clip_fs_path)
                    clip_url = get_web_url_from_fs_path(clip_fs_path)
                except Exception as e:
                    logger.error(f"Failed to generate clip for {original_video_fs_path} at {frame_timestamp_s}s: {e}", exc_info=True)
                    # Don't fail the whole search, just this clip
            else:
                clip_url = get_web_url_from_fs_path(clip_fs_path)
                logger.info(f"Clip already exists for {frame_fs_path}: {clip_url}")

            final_results.append(SearchVideoResultItem(
                video_url=get_web_url_from_fs_path(original_video_fs_path),
                frame_url=get_web_url_from_fs_path(frame_fs_path),
                frame_timestamp_s=frame_timestamp_s,
                clip_url=clip_url,
                distance=distance
            ))
        
        if not final_results:
            return SearchVideoResponse(
                message=f"No video results found for query: {query_input}.",
                results=[],
                query_type=query_type,
                query_input=query_input
            )

        return SearchVideoResponse(
            message=f"Video search completed for query: {query_input}.",
            results=final_results,
            query_type=query_type,
            query_input=query_input
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during video search: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during video search: {e}")
    finally:
        if temp_query_image_path and os.path.exists(temp_query_image_path):
            try:
                os.remove(temp_query_image_path)
                logger.info(f"Removed temporary query image file: {temp_query_image_path}")
            except Exception as e:
                logger.error(f"Failed to remove leftover temp query image file {temp_query_image_path}: {e}")

