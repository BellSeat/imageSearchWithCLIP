import os
import numpy as np
import tempfile
import pickle
import json
from vector_database import VectorDatabase

def test_vector_database():
    # Create a temporary directory for isolated testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy config.json with paths inside the temporary directory
        config = {
            "index": {
                "INDEX_PATH": os.path.join(tmpdir, "test_index.index"),
                "META_PATH": os.path.join(tmpdir, "test_meta.pkl"),
                "IMAGE_DIR": tmpdir,
                "VECTOR_DIM": 4,
                "DEVICE": "cpu",
                "INDEX_TYPE": "Flat",     # Use simple Flat index for test
                "INDEX_NLIST": 10,
                "INDEX_NPROBE": 2,
                "INDEX_M": 8,
                "INDEX_NORM": "L2",
                "INDEX_NORM_TYPE": "L2",
                "INDEX_NORM_DIM": 4
            }
        }

        # Write config to file
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Initialize the database using the config file
        db = VectorDatabase(config_path=config_path)

        # Prepare test embeddings and metadata
        vectors = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ], dtype='float32')
        metadata = ["a.jpg", "b.jpg", "c.jpg"]

        # Create the FAISS index with initial vectors
        db.create_index(vectors, metadata)
        assert db.get_total_count() == 3

        # Search using a known vector
        results = db.search([0.1, 0.2, 0.3, 0.4], top_k=2)
        assert len(results) == 2
        print("[PASS] Search returns results:", results)

        # Add a new vector and metadata
        new_vectors = np.array([
            [1.1, 1.2, 1.3, 1.4],
        ], dtype='float32')
        new_metadata = ["d.jpg"]
        db.add_vectors(new_vectors, new_metadata)
        assert db.get_total_count() == 4

        # Save and reload the database
        db.save()
        db2 = VectorDatabase(config_path=config_path)
        db2.load()
        assert db2.get_total_count() == 4
        assert db2.has_path("b.jpg")

        print("[✅ TEST COMPLETED SUCCESSFULLY]")

if __name__ == "__main__":
    test_vector_database()
