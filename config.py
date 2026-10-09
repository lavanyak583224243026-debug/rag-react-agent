import os
import sys
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env in the project root
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Configuration values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()

PDF_PATH_STR = os.getenv("PDF_PATH", "rag_file.pdf").strip()
PDF_PATH = BASE_DIR / PDF_PATH_STR

# Cache directory for vector embeddings
CACHE_DIR = BASE_DIR / ".rag_cache"


def validate_api_key() -> str:
    """Validates that a usable Gemini API key is configured.

    Returns the key or exits with a clear actionable instruction.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "YOUR_GEMINI_API_KEY_HERE":
        print("\n" + "=" * 60)
        print("❌ ERROR: Gemini API key is missing or set to placeholder!")
        print("=" * 60)
        print("Please open the `.env` file in the project folder:")
        print(f"  {BASE_DIR / '.env'}")
        print("And set your valid Gemini API key:")
        print("  GEMINI_API_KEY=AIzaSy...")
        print("=" * 60 + "\n")
        return ""
    return key
