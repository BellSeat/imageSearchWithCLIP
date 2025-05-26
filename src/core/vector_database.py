import os
import json
import pickle
import numpy as np
import faiss
from .log_util import setup_local_logger

# Logger is already set up correctly at the module level.
logger = setup_local_logger(
    log_path="logs/vector_db.log", logger_name="vector_db")


class VectorDatabase:
    def __init__(self, config_path=None):
        if config_path is None:
            raise ValueError("Config path must be provided.")

        with open(config_path, 'r') as f:
            config = json.load(f)

        index_cfg = config.get("index", {})
        self.index_path = index_cfg["INDEX_PATH"]
        self.metadata_path = index_cfg["META_PATH"]
        self.vector_dim = index_cfg.get("VECTOR_DIM", 512)
        self.index_type = index_cfg.get("INDEX_TYPE", "Flat")
        self.nlist = index_cfg.get("INDEX_NLIST", 100)
        self.nprobe = index_cfg.get("INDEX_NPROBE", 10)

        self.index = None
        self.metadata = []

    def create_index(self, vectors, metadata):
        vectors = np.array(vectors).astype('float32')
        assert vectors.shape[1] == self.vector_dim, "Vector dimension mismatch"

        if self.index_type.upper() == "IVF":
            quantizer = faiss.IndexFlatL2(self.vector_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.vector_dim, self.nlist, faiss.METRIC_L2)
            index.train(vectors)
            index.nprobe = self.nprobe
        else:
            index = faiss.IndexFlatL2(self.vector_dim)

        index.add(vectors)
        self.index = index
        self.metadata = metadata
        self.save()
        # Replaced print with logger.info
        logger.info("Created index with %d vectors.", len(metadata))

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            # Replaced print with logger.warning, as this is a recoverable condition
            logger.warning("Index or metadata file not found at %s or %s.",
                           self.index_path, self.metadata_path)
            logger.info("Creating a new, empty index and metadata file.")
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.metadata = []
            # This call will create directories if they don't exist.
            self.save()
            return  # Exit the load function after creating a new index

        self.index = faiss.read_index(self.index_path)
        # Note: 'rb' is correct for pickle
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Replaced print with logger.info
        logger.info("Loaded index from %s, metadata entries: %d",
                    self.index_path, len(self.metadata))

    def save(self):
        if self.index is None or self.metadata is None:
            raise ValueError("Index or metadata not initialized.")

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        # Note: 'wb' is correct for pickle
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        # Replaced print with logger.info
        logger.info("Index and metadata saved successfully.")

    def add_vectors(self, vectors, metadata):
        if self.index is None:
            raise ValueError("Index not initialized.")

        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)
        self.metadata.extend(metadata)
        # Replaced print with logger.info
        logger.info("Added %d new vectors to the index.", len(metadata))

    def search(self, query_vector, top_k=5):
        if self.index is None:
            raise ValueError("Index not loaded.")

        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)
        results = [(self.metadata[i], float(distances[0][j]))
                   for j, i in enumerate(indices[0]) if i < len(self.metadata)]
        return results

    def has_path(self, path):
        # This check might be slow if metadata is very large. Consider a more efficient lookup if needed.
        for item in self.metadata:
            if isinstance(item, dict) and item.get('path') == path:
                return True
        return False

    def get_total_count(self):
        if self.index is None:
            raise ValueError("Index not initialized.")
        return self.index.ntotal
