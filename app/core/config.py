from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_REFINER_MODEL: str = "codellama:7b-instruct"
    OLLAMA_HELPER_MODEL: str = "codellama:7b-instruct"

    # Vector Store
    FAISS_INDEX_PATH: Path = Path("data/faiss_index")
    EMBEDDINGS_MODEL_NAME: str = "text-embedding-3-small"

    CRAG_RELEVANCE_THRESHOLD_HIGH: float = Field(0.7, ge=0.0, le=1.0, description="Score above this -> use retrieved doc directly")
    CRAG_RELEVANCE_THRESHOLD_LOW: float = Field(0.3, ge=0.0, le=1.0, description="Score below this -> use web search only")

    # Application settings
    APP_NAME: str = "Code RAG Service with CRAG"
    LOG_LEVEL: str = "INFO"

    # Text splitting defaults for code
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

if settings.OLLAMA_HELPER_MODEL == settings.OLLAMA_REFINER_MODEL:
     print(f"INFO: Using the same Ollama model '{settings.OLLAMA_REFINER_MODEL}' for both refining and helper tasks.")
else:
     print(f"INFO: Using '{settings.OLLAMA_REFINER_MODEL}' for refining and '{settings.OLLAMA_HELPER_MODEL}' for helper tasks.")
