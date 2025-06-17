"""
Configuration management for the Intelligent Document Q&A System
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Keys
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    
    # Database Configuration
    chroma_persist_directory: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field("document_qa_collection", env="CHROMA_COLLECTION_NAME")
    
    # API Configuration
    api_host: str = Field("localhost", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(True, env="DEBUG")
    
    # Memory Configuration
    max_short_term_memory: int = Field(20, env="MAX_SHORT_TERM_MEMORY")
    max_long_term_memory: int = Field(1000, env="MAX_LONG_TERM_MEMORY")
    episodic_memory_ttl: int = Field(604800, env="EPISODIC_MEMORY_TTL")  # 7 days
    
    # Performance Configuration
    max_chunk_size: int = Field(1000, env="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 hour
    
    # Gemini Configuration
    gemini_model: str = Field("gemini-pro", env="GEMINI_MODEL")
    gemini_embedding_model: str = Field("models/embedding-001", env="GEMINI_EMBEDDING_MODEL")
    gemini_temperature: float = Field(0.1, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(2048, env="GEMINI_MAX_TOKENS")    
    # Evaluation Configuration
    eval_batch_size: int = Field(10, env="EVAL_BATCH_SIZE")
    f1_threshold: float = Field(0.7, env="F1_THRESHOLD")
    response_time_threshold: float = Field(2.0, env="RESPONSE_TIME_THRESHOLD")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }

# Global settings instance
settings = Settings()

# Create data directories if they don't exist
def ensure_data_directories():
    """Ensure all required data directories exist"""
    directories = [
        settings.chroma_persist_directory,
        "./data/documents",
        "./data/cache",
        "./data/logs",
        "./data/evaluation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize directories
ensure_data_directories()
