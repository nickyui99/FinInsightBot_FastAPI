from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path
import os
from dotenv import load_dotenv


def _resolve_chroma_dir() -> str:
    """
    Resolve the Chroma persist directory robustly regardless of current working directory.

    Priority:
    1) CHROMA_DB_DIR env var (must contain chroma.sqlite3)
    2) app/chroma_db relative to this file (preferred; matches ingestion script)
    3) chroma_db at repository root (fallback)
    """
    # 1) Environment override
    env_dir = os.getenv("CHROMA_DB_DIR")
    if env_dir:
        sqlite_path = Path(env_dir) / "chroma.sqlite3"
        if sqlite_path.exists():
            return str(Path(env_dir))

    # 2) app/chroma_db relative to this file (utils -> app)
    app_dir = Path(__file__).resolve().parent.parent  # .../app
    candidate1 = app_dir / "chroma_db"
    if (candidate1 / "chroma.sqlite3").exists():
        return str(candidate1)

    # 3) chroma_db at repo root (two levels up from utils -> repo)
    repo_root = app_dir.parent
    candidate2 = repo_root / "chroma_db"
    if (candidate2 / "chroma.sqlite3").exists():
        return str(candidate2)

    # If nothing found, raise a clear error
    raise FileNotFoundError(
        "Chroma DB not found. Checked CHROMA_DB_DIR, app/chroma_db, and chroma_db at repo root. "
        "Run the ingester (python app/scripts/ingest.py) or set CHROMA_DB_DIR."
    )


def get_retriever(k: int = 4, *, search_type: str | None = None, search_kwargs: dict | None = None):
    """Initialize and return a Chroma retriever with Google embeddings.

    Args:
        k: number of documents to retrieve (used if search_kwargs doesn't override)
        search_type: retrieval strategy (e.g., "similarity", "mmr")
        search_kwargs: kwargs passed to as_retriever (e.g., {"k": 6, "fetch_k": 30, "filter": {...}})
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    # Attempt to load .env if key not present; target app/.env relative to this file
    if not api_key:
        app_dir = Path(__file__).resolve().parent.parent
        dotenv_path = app_dir / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set in the environment")

    # Use env to override collection if needed; default matches ingest script
    collection_name = os.getenv("CHROMA_COLLECTION", "sec_filings")

    embedding = GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
        google_api_key=api_key,
    )

    persist_dir = _resolve_chroma_dir()

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name=collection_name,
    )
    # Prepare retriever configuration
    rk = {"k": k}
    if search_kwargs:
        rk.update(search_kwargs)
    # If search_type provided, pass it through; otherwise default to similarity
    if search_type:
        return vectorstore.as_retriever(search_type=search_type, search_kwargs=rk)
    return vectorstore.as_retriever(search_kwargs=rk)