from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration loaded from environment variables or .env file.

    Pydantic validates types at import time and raises a clear ValueError
    when a required variable is missing, rather than failing later at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"

    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768

    pinecone_api_key: str
    pinecone_index_name: str = "rag-chatbot"

    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rag-chatbot"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    data_folder: str = "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    lru_cache means the .env file is read exactly once per process.
    Every module calls get_settings() instead of reading env vars directly,
    keeping configuration in one place and making tests easy to override.
    """
    return Settings()
