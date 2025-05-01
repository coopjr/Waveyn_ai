import os

class Settings:
    # Embedding model and dimensions
    MODEL_NAME = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 384))
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

    # HNSW index parameters
    HNSW_INDEX_PATH = os.getenv('HNSW_INDEX_PATH', './data/hnsw_index.bin')
    ID_MAP_PATH = os.getenv('ID_MAP_PATH', './data/id_map.pkl')
    HNSW_MAPPING_PATH = os.getenv('HNSW_MAPPING_PATH', './data/hnsw_id_mapping.json')  # ✅ Added line
    MAX_ELEMENTS = int(os.getenv('MAX_ELEMENTS', 10000))
    M = int(os.getenv('HNSW_M', 16))
    EF_CONSTRUCTION = int(os.getenv('HNSW_EF_CONSTRUCTION', 32))
    EF_SEARCH = int(os.getenv('HNSW_EF_SEARCH', 32))

    # MongoDB settings
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'waveyn_db')
    MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'advisors')

    # API settings
    HOST = os.getenv('API_HOST', '0.0.0.0')
    PORT = int(os.getenv('API_PORT', 8000))
    TOP_K_DEFAULT = int(os.getenv('TOP_K_DEFAULT', 5))

# Optional cleanup of corrupted index
if os.path.exists(Settings.HNSW_INDEX_PATH):
    os.remove(Settings.HNSW_INDEX_PATH)
