# index_manager.py

import os
import pickle
import threading
import json
from typing import List

import hnswlib
import numpy as np
from config import Settings


class HnswIndex:
    def __init__(self):
        self.dim = Settings.EMBEDDING_DIM
        self.max_elements = Settings.MAX_ELEMENTS
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.lock = threading.Lock()
        self.id_map = {}

        if os.path.exists(Settings.HNSW_INDEX_PATH) and os.path.exists(Settings.ID_MAP_PATH):
            self._load()
        else:
            print("[INFO] Index files not found. Rebuilding index from scratch...")
            self._rebuild_index()

    def _initialize_index(self):
        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=Settings.EF_CONSTRUCTION,
            M=Settings.M,
            allow_replace_deleted=True
        )
        self.index.set_ef(Settings.EF_SEARCH)
        self.id_map = {}

    def _save(self):
        os.makedirs(os.path.dirname(Settings.HNSW_INDEX_PATH), exist_ok=True)
        self.index.save_index(Settings.HNSW_INDEX_PATH)
        with open(Settings.ID_MAP_PATH, 'wb') as f:
            pickle.dump(self.id_map, f)

    def _load(self):
        try:
            self.index.load_index(
                Settings.HNSW_INDEX_PATH,
                max_elements=self.max_elements,
                allow_replace_deleted=True
            )
            self.index.set_ef(Settings.EF_SEARCH)

            with open(Settings.ID_MAP_PATH, 'rb') as f:
                self.id_map = pickle.load(f)

            print("[INFO] HNSW index and ID map loaded successfully.")

            if self.index.get_current_count() == 0:
                raise RuntimeError("Index is empty after loading. Triggering rebuild...")

        except Exception as e:
            print(f"[WARN] Failed to load index or ID map: {e}")
            print("[INFO] Rebuilding index from MongoDB...")
            self._rebuild_index()

    def _rebuild_index(self):
        import pymongo
        from sentence_transformers import SentenceTransformer

        client = pymongo.MongoClient(Settings.MONGO_URI)
        db = client[Settings.MONGO_DB_NAME]
        collection = db[Settings.MONGO_COLLECTION_NAME]
        advisors = collection.find()

        texts = []
        mongo_ids = []

        for advisor in advisors:
            if "university" in advisor:
                texts.append(advisor["university"])
                mongo_ids.append(str(advisor["_id"]))

        if not texts:
            raise ValueError("No advisor documents with 'university' field found in MongoDB.")

        print(f"[INFO] Rebuilding index with {len(texts)} items...")

        model = SentenceTransformer(Settings.EMBEDDING_MODEL)
        embeddings = model.encode(texts, show_progress_bar=True)

        self._initialize_index()
        self.index.add_items(embeddings, list(range(len(mongo_ids))))
        self.id_map = {i: mongo_ids[i] for i in range(len(mongo_ids))}

        self._save()

        with open(Settings.HNSW_MAPPING_PATH, "w") as f:
            json.dump(self.id_map, f)

        print(f"[INFO] Rebuilt and saved HNSW index with {len(mongo_ids)} items.")

    def add_or_update(self, uid: int, vector: np.ndarray):
        vec = vector / np.linalg.norm(vector)
        with self.lock:
            self.index.add_items(vec.reshape(1, -1), np.array([uid]), replace_deleted=True)
            self.id_map[uid] = uid  # or associate with a real Mongo ID if needed
            self._save()

    def delete(self, uid: int):
        with self.lock:
            if uid in self.id_map:
                self.index.mark_deleted(uid)
                del self.id_map[uid]
                self._save()
            else:
                raise KeyError(f"ID {uid} not found in index.")

    def search(self, query_vec: np.ndarray, k: int):
        qvec = query_vec / np.linalg.norm(query_vec)

        with self.lock:
            current_count = self.index.get_current_count()
            if current_count < k:
                raise ValueError(f"Not enough items in index to perform top-{k} search. Only {current_count} items available.")

            labels, distances = self.index.knn_query(qvec.reshape(1, -1), k=k)

        results = []
        for label, dist in zip(labels[0], distances[0]):
            if label == -1:
                continue
            score = 1 - dist
            mongo_id = self.id_map.get(label, None)
            results.append({'id': mongo_id, 'score': float(score)})
        return results

    def embed(self, vector: np.ndarray):
        return vector / np.linalg.norm(vector)
