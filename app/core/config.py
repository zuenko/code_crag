from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CODE_MODEL: str = "codellama:7b-instruct" # Default model for code tasks

    # Vector Store
    FAISS_INDEX_PATH: Path = Path("data/faiss_index")
    EMBEDDINGS_MODEL_NAME: str = "text-embedding-3-small" # Model for creating embeddings

    # Application settings
    APP_NAME: str = "Code RAG Service"
    LOG_LEVEL: str = "INFO"

    # Text splitting defaults for code
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
