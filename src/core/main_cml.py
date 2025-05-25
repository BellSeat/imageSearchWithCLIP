import argparse
import json
import os
import sys

# Workaround for OpenMP runtime error (OMP: Error #15)
# This allows multiple OpenMP runtimes to be loaded, which can happen with
# libraries like NumPy, SciPy, or FAISS that link against MKL or OpenBLAS.
# While not a fundamental fix, it often resolves the issue in development.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add the parent directory to the system path to allow importing modules from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.embedding import Embedding
from core.vector_database import VectorDatabase 
# import core.preprocessForVideo as preprocess # Uncomment if video processing is needed later

import numpy as np
import pickle

print("[INFO] Starting main_cml.py")
print("[INFO] Current working directory:", os.getcwd())

def main():

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(CURRENT_DIR, "..", "config", "config.json")
    DEFUALT_IMAGE_PATH = os.path.join(CURRENT_DIR, "..", "database", "raw", "video", "frames", "sample_2025-05-16_01-06-10_frames_1200", "frame_0004_0-00-04-504500.jpg")
    parser = argparse.ArgumentParser(description="Process images/video and interact with vector database.")
    parser.add_argument("--image", type=str, help="Path to a single image file to embed and add to DB.")
    parser.add_argument("--image_folder", type=str, help="Path to a folder containing images to embed and add to DB.", default=DEFUALT_IMAGE_PATH)
    parser.add_argument("--text", type=str, help="Text description (optional) for single image.", default="")
    parser.add_argument("--search_text", type=str, help="Text query to search for in the database.", default="")
    parser.add_argument("--search_image", type=str, help="Path to an image file to use as a query for searching the database.", default=DEFUALT_IMAGE_PATH)
    parser.add_argument("--config", type=str, help="Path to config.json", default=CONFIG_PATH)
    
    # parser.add_argument("--video", type=str, help="Path to video file to process.") # Uncomment if video processing is needed later
    
    args = parser.parse_args()

    # Initialize embedding and DB
    try:
        embedder = Embedding(config_path=args.config)
        vectordb = VectorDatabase(config_path=args.config)
        vectordb.load() # Load existing index or create a new one if not found
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to initialize Embedding or VectorDatabase: {e}")
        sys.exit(1) # Exit if core components cannot be initialized

    # Process a single image if provided
    if args.image:
        print(f"[INFO] Processing single image: {args.image}")
        metadata = {"path": args.image, "text": args.text}
        image_embedding = embedder.embed_image(args.image)
        if image_embedding is not None:
            vectordb.add_vectors([image_embedding], [metadata])
            vectordb.save()
            print("[✅] Single image embedded and added to database.")
        else:
            print("[❌] Failed to embed single image.")

    # Process images from a folder if provided
    if args.image_folder:
        if os.path.exists(args.image_folder):
            print(f"[INFO] Processing images from folder: {args.image_folder}")
            batch_embeddings = []
            batch_metadata = []
            processed_count = 0
            failed_count = 0

            for filename in os.listdir(args.image_folder):
                # Check for common image file extensions
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    image_path = os.path.join(args.image_folder, filename)
                    print(f"[INFO] Attempting to embed: {image_path}")
                    image_embedding = embedder.embed_image(image_path)
                    if image_embedding is not None:
                        # Create specific metadata for each image in the folder
                        # You might want to extract text from image_path or use a default
                        image_metadata = {"path": image_path, "text": f"Image from {args.image_folder}"} 
                        batch_embeddings.append(image_embedding)
                        batch_metadata.append(image_metadata)
                        processed_count += 1
                    else:
                        print(f"[❌] Failed to embed image: {filename}")
                        failed_count += 1
            
            if batch_embeddings:
                print(f"[INFO] Adding {len(batch_embeddings)} embeddings to database in batch...")
                vectordb.add_vectors(batch_embeddings, batch_metadata)
                vectordb.save() # Save only once after all images in the folder are processed
                print(f"[✅] Successfully processed {processed_count} images from folder and added to database.")
            else:
                print(f"[ℹ️] No images were successfully embedded from folder: {args.image_folder}")
            
            if failed_count > 0:
                print(f"[⚠️] Failed to process {failed_count} images from folder.")

        else:
            print(f"[❌] Image folder not found: {args.image_folder}")

    # Process text embedding (if only text is provided, not for search specifically)
    # This block is for generating an embedding for a standalone text, not for searching.
    if args.text and not args.image and not args.image_folder and not args.search_text and not args.search_image: 
        print(f"[INFO] Generating embedding for standalone text: \"{args.text}\"")
        text_embedding = embedder.embed_text(args.text)
        if text_embedding is not None:
            print("[✅] Text embedding sample (first 5 dims):", text_embedding[:5])
        else:
            print("[❌] Failed to embed text.")

    # Perform search if search_text is provided
    if args.search_text:
        print(f"[INFO] Performing search for text query: \"{args.search_text}\"")
        query_embedding = embedder.embed_text(args.search_text)
        if query_embedding is not None:
            # Perform search, adjust top_k as needed
            search_results = vectordb.search(query_embedding, top_k=5) 
            if search_results:
                print(f"[✅] Search results for \"{args.search_text}\":")
                for path, distance in search_results:
                    print(f"  - Path: {path}, Distance: {distance:.4f}")
            else:
                print(f"[ℹ️] No results found for \"{args.search_text}\". Please ensure the database contains relevant embeddings.")
        else:
            print("[❌] Failed to create embedding for search text query. Is the text valid?")

    # Perform search if search_image is provided
    if args.search_image:
        if os.path.exists(args.search_image):
            print(f"[INFO] Performing search for image query: \"{args.search_image}\"")
            query_embedding = embedder.embed_image(args.search_image)
            if query_embedding is not None:
                # Perform search, adjust top_k as needed
                search_results = vectordb.search(query_embedding, top_k=5) 
                if search_results:
                    print(f"[✅] Search results for image \"{args.search_image}\":")
                    for path, distance in search_results:
                        print(f"  - Path: {path}, Distance: {distance:.4f}")
                else:
                    print(f"[ℹ️] No results found for image \"{args.search_image}\". Please ensure the database contains relevant embeddings.")
            else:
                print("[❌] Failed to create embedding for search image query. Is the image path valid?")
        else:
            print(f"[❌] Search image file not found: {args.search_image}")


    # Video processing (uncomment and fix if needed)
    # if args.video:
    #     video_path = args.video
    #     if os.path.exists(video_path):
    #         print(f"[INFO] Processing video: {video_path}")
    #         # You'll need to ensure preprocess.extract_keyframmes returns paths
    #         # and that build_output_dir is called to get the output folder.
    #         # output_dir = preprocess.build_output_dir(video_path)
    #         # frames = preprocess.extract_keyframmes(video_path, output_dir)
    #         # for frame_path in frames: # Iterate over paths, not frame objects
    #         #     image_embedding = embedder.embed_image(frame_path)
    #         #     if image_embedding is not None:
    #         #         # Create metadata specific to the video frame
    #         #         frame_metadata = {"path": frame_path, "text": f"Frame from {video_path}"}
    #         #         vectordb.add_vectors([image_embedding], [frame_metadata])
    #         #     else:
    #         #         print(f"[❌] Failed to embed frame: {frame_path}")
    #         # vectordb.save()
    #         # print("[✅] Video processed and embeddings added to database.")
    #     else:
    #         print(f"[❌] Video file not found: {video_path}")

if __name__ == "__main__":
    main()
