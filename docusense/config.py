import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DOCUSENSE_DATA_DIR", BASE_DIR / "data"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # characters
TOP_K = int(os.getenv("TOP_K", "5"))
