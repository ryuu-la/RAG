from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH, override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    google_api_key: str = ""
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


settings = Settings()
