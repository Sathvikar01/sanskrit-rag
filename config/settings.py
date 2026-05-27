"""Configuration settings for SansRAG project."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDEA_API_KEY", "")
NVIDEA_API_KEY = NVIDIA_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

NVIDIA_API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/embeddings")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

BYT5_MODEL = os.getenv("BYT5_MODEL", "google/byt5-small")
BYT5_SANSKRIT_MODEL = os.getenv("BYT5_SANSKRIT_MODEL", "google/byt5-small")

DENSE_DIM = 1024
COLBERT_DIM = 128
MAX_TEXT_LENGTH = 8000
MAX_CHUNKS_FOR_TESTING = 0

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 256
IVF_NLIST = 1024
SPARSE_DROP_RATIO = 0.2

COLLECTION_NAMES = {
    "raw": "sansr_raw",
    "lemma_morph": "sansr_lemma_morph",
    "seg_lemma": "sansr_seg_lemma"
}

QDRANT_COLLECTION = "sansr_seg_lemma"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

L1_REG_LAMBDA = 0.01
L2_REG_LAMBDA = 0.001

RRF_K = 20
RRF_TOP_K = 10
MIN_RRF_SCORE = 0.005

RRF_WEIGHTS_VERSE_FILTER = {"neo4j": 0.7, "qdrant": 0.3}
RRF_WEIGHTS_NO_FILTER = {"neo4j": 0.4, "qdrant": 0.6}

SEMANTIC_CONSISTENCY_THRESHOLD = 0.30
MAX_CONSISTENCY_RETRIES = 2

TEST_QUERIES = [
    "dharma",
    "karma",
    "mokṣa",
    "yoga",
    "arjuna",
    "kṛṣṇa",
    "bhakti",
    "dhṛtarāṣṭra uvāca",
    "dharma-kṣetre kuru-kṣetre",
]
