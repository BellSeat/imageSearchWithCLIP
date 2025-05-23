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

# Workaround for OpenMP runtime error (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add the parent directory to the system path to allow importing modules from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.embedding import Embedding
from core.vector_database import VectorDatabase
from core.log_util import setup_local_logger

logger = setup_local_logger(log_path="logs/api_run.log", logger_name="fastapi_app")

# --- Configuration Paths ---
SCRIPT_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(SCRIPT_DIR,"..", "config", "config.json")

UPLOAD_TEMP_DIR = os.path.join(SCRIPT_DIR, "temp_uploads") 

STATIC_CONTENT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "database"))
FRAMES_DIR = os.path.join(STATIC_CONTENT_ROOT, "raw", "video", "frames")
UPLOADED_IMAGES_DIR = os.path.join(STATIC_CONTENT_ROOT, "uploaded_images")

os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)

# --- Global Instances ---
embedder: Optional[Embedding] = None
vectordb: Optional[VectorDatabase] = None

app = FastAPI(
    title="Image and Text Search API",
    description="API for embedding images/text and performing similarity searches using CLIP and FAISS.",
    version="1.0.0"
)

# --- Pydantic Models ---
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
# This function MUST be defined at the top-level of the script,
# before any endpoint functions that call it.
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
    Converts a server-side file system path to a web-accessible URL based on static mounts.
    """
    if fs_path.startswith(FRAMES_DIR):
        relative_path = os.path.relpath(fs_path, FRAMES_DIR).replace("\\", "/")
        return f"/static/frames/{relative_path}"
    elif fs_path.startswith(UPLOADED_IMAGES_DIR):
        relative_path = os.path.relpath(fs_path, UPLOADED_IMAGES_DIR).replace("\\", "/")
        return f"/static/uploads/{relative_path}"
    else:
        logger.warning(f"Could not determine static URL for path: {fs_path}. Returning raw path.")
        return fs_path


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
