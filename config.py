import os
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# OpenAI
# =========================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# =========================================================
# Embedding Model
# =========================================================

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMENSIONS = 768

# =========================================================
# ChromaDB
# =========================================================

CHROMA_PERSIST_DIR = "./chroma_db_nomic"
CHROMA_COLLECTION_NAME = "rag_documents"

# =========================================================
# Chunking
# =========================================================

CHUNK_MAX_CHARACTERS = 1200
CHUNK_OVERLAP = 150

# =========================================================
# Embedding batching
# =========================================================

EMBED_BATCH_SIZE = 100

# =========================================================
# Retrieval
# =========================================================

DEFAULT_TOP_K = 5

# =========================================================
# Refusal
# =========================================================

REFUSAL_THRESHOLD = 0.65
RELEVANT_THRESHOLD = 0.60
MIN_RELEVANT_HITS = 1

REFUSAL_MESSAGE = (
    "I don't have enough information in the provided documents "
    "to accurately answer your question."
)