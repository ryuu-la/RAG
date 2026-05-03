from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Locate .env ──────────────────────────────────────────
# Priority: backend/.env → project-root/.env
# This way users can place .env in EITHER location.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _BACKEND_DIR.parent

_ENV_PATH = _BACKEND_DIR / ".env"
if not _ENV_PATH.exists():
    _root_env = _PROJECT_ROOT / ".env"
    if _root_env.exists():
        _ENV_PATH = _root_env

load_dotenv(_ENV_PATH, override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    google_api_key: str = ""
    openrouter_api_key: str = ""
    default_model: str = "gemma-4-31b-it"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 6
    chunk_size: int = 1400
    chunk_overlap: int = 200
    model_cache_dir: str = "./data/models"
    chroma_collection: str = "rag_chunks"
    max_context_tokens: int = 5000
    ocr_enabled: bool = True
    ocr_min_chars_per_page: int = 40

    data_dir: str = "./data"
    upload_dir: str = "./data/uploads"
    chroma_dir: str = "./data/chroma"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)

    @property
    def model_cache_path(self) -> Path:
        return Path(self.model_cache_dir)

    @property
    def export_path(self) -> Path:
        return self.data_path / "exports"

    @property
    def has_api_key(self) -> bool:
        return bool(self.google_api_key and self.google_api_key.strip())


settings = Settings()


# ── Startup banner (runs once on import) ─────────────────
def _startup_check() -> None:
    print("\n" + "=" * 56)
    print("  RAG Studio — Backend")
    print("=" * 56)
    print(f"  .env loaded from : {_ENV_PATH}")
    if settings.has_api_key:
        masked = settings.google_api_key[:8] + "..." + settings.google_api_key[-4:]
        print(f"  GOOGLE_API_KEY   : {masked}")
    else:
        print("  GOOGLE_API_KEY   : ❌ NOT SET")
        
    if settings.openrouter_api_key:
        masked_or = settings.openrouter_api_key[:8] + "..." + settings.openrouter_api_key[-4:]
        print(f"  OPENROUTER_KEY   : {masked_or}")
    else:
        print("  OPENROUTER_KEY   : ❌ NOT SET")
        print()
        print("  ⚠  No API key found! The chat will not work.")
        print(f"  ➜  Add your key to: {_ENV_PATH}")
        print("     GOOGLE_API_KEY=your_key_here")
        print()
        print("  Get a free key at: https://aistudio.google.com/apikey")
    print(f"  Default model    : {settings.default_model}")
    print(f"  Embedding model  : {settings.embedding_model}")
    print("=" * 56 + "\n")


_startup_check()
