# this script is used to create a class for embedding image and text to vector
# by using CLIP
import os
import json
import clip
import numpy as np
import torch
from PIL import Image
from torchvision import transforms



class Embedding:
    def __init__(self, config_path="config/config.json"):
        with open(config_path, 'r') as f:
            config = json.load(f)
        clip_cfg = config.get("CLIP", {})

        self.device = clip_cfg.get("DEVICE", "cpu")
        self.model_name = clip_cfg.get("MODEL_NAME", "ViT-B/32")
        self.model, self.preporcess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        print(f"[INFO] CLIP model loaded: {self.model_name} on {self.device}")
   
    def embed_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preporcess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.cpu().numpy()
                return embedding.flatten()
        except Exception as e:
            print(f"[ERROR] Error embedding image {image_path}: {e}")
            return None
        
    def embed_text(self, text):
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_text(text_tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"[ERROR] Failed to process text \"{text}\": {e}")
            return None
        
    def embed_image_and_text(self, image_path, text):
        image_embedding = self.embed_image(image_path)
        text_embedding = self.embed_text(text)
        return image_embedding, text_embedding
    
    
        