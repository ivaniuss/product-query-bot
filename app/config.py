from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    embedding_model: str
    chat_model: str
    top_k: int
    temperature: float
    max_tokens: int
    vector_store_path: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        env_prefix = ''

@lru_cache()
def get_settings():
    return Settings()
