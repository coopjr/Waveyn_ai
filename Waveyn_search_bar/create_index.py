# create_index.py
import hnswlib
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model loaded: 'sentence-transformers/all-MiniLM-L6-v2'")
# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["waveyn_db"]
collection = db["advisors"]
print("✅ Connected to MongoDB database 'waveyn' and collection 'advisors'")
# Load data
advisors = list(collection.find({"university": {"$exists": True}}, {"_id": 1, "university": 1}))
texts = [advisor["university"] for advisor in advisors]
ids = [i for i in range(len(texts))]
print(f"✅ Loaded {len(advisors)} advisors with university data from MongoDB")
# Filter out empty university names
print(f"Fetched {len(texts)} university entries from MongoDB.")
print(texts)


# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
print("✅ Generated embeddings for university names")
# Initialize HNSW index
dim = embeddings.shape[1]
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
index.add_items(embeddings, ids)
print("✅ Added items to the HNSW index")

# Save index
index.save_index("my_index.bin")
print("✅ Index saved as 'my_index.bin'")
