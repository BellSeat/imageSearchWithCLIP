# This script creates a class for embedding image and text to vector using CLIP.
import os
import json
import clip
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Optional, Tuple, Any

class Embedding:
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initializes the Embedding class, loading the CLIP model based on config.
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        clip_cfg = config.get("CLIP", {})

        self.device = clip_cfg.get("DEVICE", "cpu")
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.model_name = clip_cfg.get("MODEL_NAME", "ViT-B/32")
        self.model, self.preporcess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        print(f"[INFO] CLIP model loaded: {self.model_name} on {self.device}")
   
    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Embeds a single image to a vector.
        Returns a flattened NumPy array or None if an error occurs.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preporcess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy().flatten()
        except FileNotFoundError:
            print(f"[ERROR] Image file not found: {image_path}")
            return None
        except Exception as e:
            print(f"[ERROR] Error embedding image {image_path}: {e}")
            return None
        
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Embeds a single text string to a vector.
        Returns a flattened NumPy array or None if an error occurs.
        """
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_text(text_tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"[ERROR] Failed to process text \"{text}\": {e}")
            return None
        
    def embed_image_and_text(self, image_path: str, text: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Embeds both an image and a text string.
        Returns a tuple of (image_embedding, text_embedding).
        """
        image_embedding = self.embed_image(image_path)
        text_embedding = self.embed_text(text)
        return image_embedding, text_embedding

    def batch_embed_from_json(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Batch embeds images and optional text descriptions from a JSON file.

        Args:
            json_file_path (str): Path to the JSON file containing image data.
                                  Expected format:
                                  [
                                      {"image_path": "path/to/img1.jpg", "text_description": "optional text"},
                                      {"image_path": "path/to/img2.png"}
                                  ]

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains:
                - "image_path": Original path of the image.
                - "image_embedding": NumPy array of the image embedding (or None if failed).
                - "text_description": Original text description (or None if not provided/failed).
                - "text_embedding": NumPy array of the text embedding (or None if not provided/failed).
                - "status": "success" or "failed" for the overall processing of this entry.
        """
        results = []
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"[ERROR] JSON file '{json_file_path}' does not contain a list of objects.")
                return []

            print(f"[INFO] Starting batch embedding for {len(data)} entries from '{json_file_path}'...")
            
            for i, entry in enumerate(data):
                image_path = entry.get("image_path")
                text_description = entry.get("text_description")
                
                if not image_path:
                    print(f"[WARNING] Entry {i} in JSON has no 'image_path'. Skipping.")
                    continue

                image_embedding = None
                text_embedding = None
                entry_status = "failed"

                # Embed Image
                img_embed = self.embed_image(image_path)
                if img_embed is not None:
                    image_embedding = img_embed
                    entry_status = "success" 

                # Embed Text (if provided)
                if text_description:
                    txt_embed = self.embed_text(text_description)
                    if txt_embed is not None:
                        text_embedding = txt_embed
                    else:
                        entry_status = "failed" # If text embedding fails, mark overall as failed

                # If image embedding failed, ensure overall status is failed
                if image_embedding is None:
                    entry_status = "failed"
                
                results.append({
                    "image_path": image_path,
                    "image_embedding": image_embedding,
                    "text_description": text_description,
                    "text_embedding": text_embedding,
                    "status": entry_status
                })
                print(f"[INFO] Processed entry {i+1}/{len(data)}: {image_path} (Status: {entry_status})")

            print(f"[INFO] Batch embedding completed. Processed {len(results)} entries.")
            return results

        except FileNotFoundError:
            print(f"[ERROR] JSON file not found: {json_file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format in '{json_file_path}': {e}")
            return []
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during batch embedding: {e}")
            return []
