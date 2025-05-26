# This script creates a class for embedding image and text to vector using CLIP.
import os
import json
import clip
import numpy as np
import torch
from PIL import Image
from torchvision import transforms # Still imported, but commented out in the actual usage. Consider removing if not used.
from typing import List, Dict, Optional, Tuple, Any

class Embedding:
    """
    A class to handle image and text embedding using the CLIP model.
    """
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initializes the Embedding class, loading the CLIP model based on configuration.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f: # Explicitly specify encoding
                config = json.load(f)
        except FileNotFoundError:
            print(f"[CRITICAL ERROR] Config file not found: {config_path}")
            raise # Re-raise to prevent initialization if config is missing
        except json.JSONDecodeError as e:
            print(f"[CRITICAL ERROR] Invalid JSON format in config file {config_path}: {e}")
            raise # Re-raise if config is malformed

        clip_cfg = config.get("CLIP", {})

        self.device = clip_cfg.get("DEVICE", "cpu")
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.model_name = clip_cfg.get("MODEL_NAME", "ViT-B/32")
        
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device) # Corrected typo: preporcess to preprocess
            self.model.eval() # Set model to evaluation mode
            print(f"[INFO] CLIP model loaded: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load CLIP model {self.model_name} on {self.device}: {e}")
            raise # Re-raise if model loading fails
   
    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Embeds a single image to a vector using the CLIP model.

        Args:
            image_path (str): The file path to the image.

        Returns:
            Optional[np.ndarray]: A flattened NumPy array representing the image embedding,
                                  or None if the image file is not found or an error occurs.
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file not found: {image_path}")
            return None
        
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device) # Corrected typo: preporcess to preprocess

            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy().flatten()
        except Exception as e: # Catching general Exception but logging specific info
            print(f"[ERROR] Error embedding image {image_path}: {e}")
            return None
        
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Embeds a single text string to a vector using the CLIP model.

        Args:
            text (str): The text string to embed.

        Returns:
            Optional[np.ndarray]: A flattened NumPy array representing the text embedding,
                                  or None if an error occurs.
        """
        if not text: # Handle empty string input
            print("[WARNING] Attempted to embed an empty text string. Returning None.")
            return None

        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_text(text_tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy().flatten()
        except Exception as e: # Catching general Exception but logging specific info
            print(f"[ERROR] Failed to process text \"{text}\": {e}")
            return None
        
    def embed_image_and_text(self, image_path: str, text: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Embeds both an image and a text string.

        Args:
            image_path (str): The file path to the image.
            text (str): The text string to embed.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing the
                                                                image embedding and text embedding.
        """
        image_embedding = self.embed_image(image_path)
        text_embedding = self.embed_text(text)
        return image_embedding, text_embedding

    def batch_embed_from_json(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Batch embeds images and optional text descriptions from a JSON file.

        The JSON file is expected to contain a list of objects, where each object has
        an "image_path" (required) and optionally a "text_description".

        Example JSON format:
        [
            {"image_path": "path/to/img1.jpg", "text_description": "A cat sitting on a couch."},
            {"image_path": "path/to/img2.png"} // Text description is optional
        ]

        Args:
            json_file_path (str): Path to the JSON file containing image data.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains:
                - "image_path": Original path of the image.
                - "image_embedding": NumPy array of the image embedding (or None if failed).
                - "text_description": Original text description (or None if not provided).
                - "text_embedding": NumPy array of the text embedding (or None if not provided or failed).
                - "status": "success" or "failed" for the overall processing of this entry.
        """
        results = []
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"[ERROR] JSON file '{json_file_path}' does not contain a list of objects. Returning empty list.")
                return []

            print(f"[INFO] Starting batch embedding for {len(data)} entries from '{json_file_path}'...")
            
            for i, entry in enumerate(data):
                image_path = entry.get("image_path")
                text_description = entry.get("text_description") # text_description can be None or empty string

                if not image_path:
                    print(f"[WARNING] Entry {i+1} in JSON has no 'image_path'. Skipping this entry.")
                    continue

                image_embedding = None
                text_embedding = None
                entry_status = "failed" # Default status is failed

                # Embed Image
                img_embed = self.embed_image(image_path)
                if img_embed is not None:
                    image_embedding = img_embed
                    entry_status = "success" # Image embedded successfully, tentatively set status to success

                # Embed Text (only if text_description is provided and not empty)
                if text_description: # This checks for non-None and non-empty strings
                    txt_embed = self.embed_text(text_description)
                    if txt_embed is not None:
                        text_embedding = txt_embed
                    else:
                        # If text embedding fails, the overall status for this entry is failed
                        entry_status = "failed" 

                # Final status check: If image embedding failed, ensure overall status is failed
                if image_embedding is None:
                    entry_status = "failed"
                
                results.append({
                    "image_path": image_path,
                    "image_embedding": image_embedding,
                    "text_description": text_description,
                    "text_embedding": text_embedding,
                    "status": entry_status
                })
                print(f"[INFO] Processed entry {i+1}/{len(data)}: '{image_path}' (Status: {entry_status})")

            print(f"[INFO] Batch embedding completed. Processed {len(results)} entries.")
            return results

        except FileNotFoundError:
            print(f"[ERROR] JSON file not found: {json_file_path}. Returning empty list.")
            return []
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format in '{json_file_path}': {e}. Returning empty list.")
            return []
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during batch embedding from JSON: {e}. Returning empty list.")
            return []

